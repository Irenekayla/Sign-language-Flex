#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_quant.h" // Include the quantized model

// Define the input and output tensors
const int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = tflite::GetModel(model_quant_tflite);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
  return;
}

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
interpreter.AllocateTensors();

TfLiteTensor* input = interpreter.input(0);

void setup() {
  Serial.begin(115200);

  // Initialize the input data
  // Fill the input tensor with your data here
  for (int i = 0; i < input->bytes; i++) {
    input->data.f[i] = 0.0f; // Replace with actual data
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
  }

  TfLiteTensor* output = interpreter.output(0);

  // Print the output
  for (int i = 0; i < output->bytes; i++) {
    Serial.print(output->data.f[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Run the model in a loop if needed
}