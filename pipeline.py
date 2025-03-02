import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, StandardOptions
from google.cloud import pubsub_v1
import json
from src.object_detection import detect_pedestrians
from src.depth_estimation import estimate_depth

class ProcessImage(beam.DoFn):
    def process(self, image_path):
        pedestrians = detect_pedestrians(image_path)
        if not pedestrians:
            return
        
        depths = estimate_depth(image_path, pedestrians)

        results = []
        for i, pedestrian in enumerate(pedestrians):
            results.append({
                "bounding_box": pedestrian["bounding_box"],
                "confidence": pedestrian["confidence"],
                "depth": depths[i]
            })
        
        yield json.dumps(results)

def run(argv=None):
    # Set up pipeline options
    options = PipelineOptions(argv)
    options.view_as(StandardOptions).streaming = True

    # Create a Beam pipeline
    with beam.Pipeline(options=options) as p:
        (p 
         | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(topic="projects/fine-cycling-451417-q7/topics/mnist_predict")
         | "Process Image" >> beam.ParDo(ProcessImage())
         | "Write to Pub/Sub" >> beam.io.WriteToPubSub(topic="projects/fine-cycling-451417-q7/topics/mnist_image"))

if __name__ == "__main__":
    run()
