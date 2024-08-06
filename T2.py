import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import Row

input_file = sys.argv[1]
output_file = sys.argv[2]

spark = SparkSession.builder.appName("FlightDelayPredictionModelTraining").getOrCreate()

print("Loading data from:", input_file)
data = spark.read.csv(input_file, header=True, inferSchema=True)
print("Data loaded successfully")

required_columns = ["Airline_index", "Origin_index", "Dest_index", "CRSDepTime", "Distance", "DepDel15"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Missing columns in data: {missing_columns}")
    sys.exit(-1)

assembler = VectorAssembler(
    inputCols=["Airline_index", "Origin_index", "Dest_index", "CRSDepTime", "Distance"],
    outputCol="features"
)

print("Creating features column")
data = assembler.transform(data)
print("Features column created successfully")

label_counts = data.groupBy("DepDel15").count().collect()
min_count = min([row["count"] for row in label_counts])

fractions = {0: min_count / label_counts[0]['count'], 1: min_count / label_counts[1]['count']}
balanced_data = data.sampleBy("DepDel15", fractions=fractions, seed=42)

train_data, test_data = balanced_data.randomSplit([0.7, 0.3], seed=42)

lr = LogisticRegression(labelCol="DepDel15", featuresCol="features", regParam=0.01)

print("Training model")
lr_model = lr.fit(train_data)
print("Model trained successfully")

print("Making predictions")
predictions = lr_model.transform(test_data)
print("Predictions made successfully")

print("Evaluating model")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="DepDel15", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

precision_evaluator = MulticlassClassificationEvaluator(labelCol="DepDel15", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print(f"Precision: {precision}")

recall_evaluator = MulticlassClassificationEvaluator(labelCol="DepDel15", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)
print(f"Recall: {recall}")

f1_evaluator = MulticlassClassificationEvaluator(labelCol="DepDel15", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score}")

results = [
    Row(Metric="Accuracy", Value=accuracy),
    Row(Metric="Precision", Value=precision),
    Row(Metric="Recall", Value=recall),
    Row(Metric="F1 Score", Value=f1_score)
]

results_df = spark.createDataFrame(results)
results_rdd = results_df.rdd.map(lambda row: f"{row['Metric']}: {row['Value']}")

results_rdd.coalesce(1).saveAsTextFile(output_file)
print(f"Evaluation results saved to {output_file}")

spark.stop()
