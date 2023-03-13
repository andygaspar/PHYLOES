import pandas as pd
import dask.dataframe as dd
import numpy as np

from Bio.Seq import Seq
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, IntegerType
import Data_.csv_.Dist.Dist.sequence_distances as sd


def get_words(seq_str, k):
    words = set()
    for i in range(len(seq_str) - k):
        words.add(seq_str[i: i + k])

    return words


def fun(row):
    res = []
    seq = Seq(row.sequence)
    for word in all_words_set:
        res.append((row.taxon, word, seq.count_overlap(word)))
    return res


word_length = 9

spark = SparkSession.builder.config('spark.driver.memory','16g').getOrCreate()

df_pandas = pd.read_csv("Data_/csv_/Dist/Dist/Covid19_sequences.csv")
ddf_pandas = dd.from_pandas(df_pandas, npartitions=4)
dask_series = ddf_pandas['sequence'].apply(get_words, args=(word_length,), meta=('x', 'int'))
ddf_pandas['word_set'] = dask_series

df_pandasf = ddf_pandas.compute()
all_words_set = set().union(*df_pandasf.word_set.to_list())



df = spark.read.csv('Data_/csv_/Dist/Dist/Covid19_sequences.csv', inferSchema=True, header=True).limit(418)
# df = df.withColumn('word_set', get_words(f.col('sequence'), f.lit(8)))

print("here *************************************")


out = df.rdd.flatMap(lambda r: fun(r))
out.toDF(["taxon", "word", "word_count"]).coalesce(1).write.csv('dati_covid_' + str(word_length), header=True)
print("done *************************************")

# df_pandas = out.toDF(["taxon", "word", "count"]).toPandas()
# df_pandas.taxon.unique().shape[0]
#
# df = pd.read_csv('dati_covid_' + str(word_length) + '/covid_freq_' + str(word_length) + '.csv')
#
# for taxon in df.taxon.unique():
#     print(df[df.taxon == taxon].shape)
#
# matrix = np.zeros((df.taxon.unique().shape[0], df.word.unique().shape[0]), dtype=np.ubyte)
# for i, taxon in enumerate(df.taxon.unique()):
#     matrix[i] = df[df.taxon == taxon].sort_values(by='word').word_count
#
#
# np.savetxt('covid' + str(word_length) + '.txt',matrix)


