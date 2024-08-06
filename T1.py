import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

Combined_flights = sys.argv[1]
Output_file = sys.argv[2]

spark = SparkSession.builder.appName("FlightDelayPrediction").getOrCreate()

try:
    print("Loading data from:", Combined_flights)
    data = spark.read.csv(Combined_flights, header=True, inferSchema=True)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(-1)

print("Dropping missing values")
data = data.na.drop()

selected_columns = ["Airline", "Origin", "Dest", "CRSDepTime", "Distance", "DepDel15"]
data = data.select(selected_columns)
print("Selected columns:", selected_columns)

indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in ["Airline", "Origin", "Dest"]]

assembler = VectorAssembler(
    inputCols=[
        "Airline_index",
        "Origin_index",
        "Dest_index",
        "CRSDepTime",
        "Distance"
    ],
    outputCol="features"
)

pipeline = Pipeline(stages=indexers + [assembler])
try:
    print("Fitting pipeline")
    data = pipeline.fit(data).transform(data)
    print("Pipeline fitted successfully")
except Exception as e:
    print(f"Error during pipeline fitting: {e}")
    sys.exit(-1)

output_columns = ["Airline_index", "Origin_index", "Dest_index", "CRSDepTime", "Distance", "DepDel15"]
final_data = data.select(output_columns)
print("Selected final columns for output")

final_data.show(5)

try:
    print("Saving data to:", Output_file)
    final_data.coalesce(1).write.csv(Output_file, header=True, mode="overwrite")
    print("Data saved successfully")
except Exception as e:
    print(f"Error saving data: {e}")
    sys.exit(-1)

spark.stop()