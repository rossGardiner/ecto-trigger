#include "main_functions.h"
#include "esp_heap_caps.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "esp_sleep.h"

#if MODEL == 1
#include "1_model_data.h"
#define MODEL_DATA __1_int8_tflite
#define MODEL_LEN __1_int8_tflite_len
#define MODEL_NAME "Model 1"
#define INPUT_HEIGHT 96
#define INPUT_WIDTH 96
#define INPUT_CHANNELS 1
#elif MODEL == 2
#include "2_model_data.h"
#define MODEL_DATA __2_int8_tflite
#define MODEL_LEN __2_int8_tflite_len
#define MODEL_NAME "Model 2"
#define INPUT_HEIGHT 96
#define INPUT_WIDTH 96
#define INPUT_CHANNELS 3
#elif MODEL == 3
#include "3_model_data.h"
#define MODEL_DATA __3_int8_tflite
#define MODEL_LEN __3_int8_tflite_len
#define MODEL_NAME "Model 3"
#define INPUT_HEIGHT 120
#define INPUT_WIDTH 160
#define INPUT_CHANNELS 1
#elif MODEL == 4
#include "4_model_data.h"
#define MODEL_DATA __4_int8_tflite
#define MODEL_LEN __4_int8_tflite_len
#define MODEL_NAME "Model 4"
#define INPUT_HEIGHT 120
#define INPUT_WIDTH 160
#define INPUT_CHANNELS 3
#elif MODEL == 5
#include "5_model_data.h"
#define MODEL_DATA __5_int8_tflite
#define MODEL_LEN __5_int8_tflite_len
#define MODEL_NAME "Model 5"
#define INPUT_HEIGHT 96
#define INPUT_WIDTH 96
#define INPUT_CHANNELS 1
#elif MODEL == 6
#include "6_model_data.h"
#define MODEL_DATA __6_int8_tflite
#define MODEL_LEN __6_int8_tflite_len
#define MODEL_NAME "Model 6"
#define INPUT_HEIGHT 96
#define INPUT_WIDTH 96
#define INPUT_CHANNELS 3
#elif MODEL == 7
#include "7_model_data.h"
#define MODEL_DATA __7_int8_tflite
#define MODEL_LEN __7_int8_tflite_len
#define MODEL_NAME "Model 7"
#define INPUT_HEIGHT 120
#define INPUT_WIDTH 160
#define INPUT_CHANNELS 1
#elif MODEL == 8
#include "8_model_data.h"
#define MODEL_DATA __8_int8_tflite
#define MODEL_LEN __8_int8_tflite_len
#define MODEL_NAME "Model 8"
#define INPUT_HEIGHT 120
#define INPUT_WIDTH 160
#define INPUT_CHANNELS 3
#else
#error "No model defined"
#endif

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int scratchBufSize = 80 * 1024;
//constexpr int kTensorArenaSize = 275 * 1024 + scratchBufSize;
const int kTensorArenaSize = 2*MODEL_LEN + scratchBufSize;
static uint8_t *tensor_arena;
static uint8_t* model_data = nullptr;
}  // namespace

void setup() {
  ESP_LOGI("MEMORY", "Internal memory available: %d bytes", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  ESP_LOGI("MEMORY", "PSRAM available: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  ESP_LOGI("INPUT", "Image input size requested: %d w, %d h, %d c", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
  
  // model_data = (uint8_t*)heap_caps_malloc(MODEL_LEN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  // if (model_data == nullptr) {
  //   printf("Failed to allocate memory for model data in PSRAM.");
  //   return;
  // }

  // // Copy model data to allocated PSRAM
  // memcpy(model_data, MODEL_DATA, MODEL_LEN);

  model = tflite::GetModel(MODEL_DATA);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    printf("Requesting memory of %d bytes\n", kTensorArenaSize);
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
}

#define TARGET_INTERVAL_US 500000

#ifndef CLI_ONLY_INFERENCE
void loop() {
  int64_t start_time = esp_timer_get_time();
  run_inference(input);
  int64_t elapsed_time = esp_timer_get_time() - start_time;
  printf("elapsed time %lld\n", elapsed_time);
  int64_t delay_time = TARGET_INTERVAL_US - elapsed_time;
  
  if (delay_time > 0) {
    printf("sleeping for %lld microsecs\n", delay_time);
    //int64_t start_sleep_time = esp_timer_get_time();
    //esp_sleep_enable_timer_wakeup(delay_time);
    //esp_light_sleep_start();
    //printf("time actually spent in sleep: %lld\n", esp_timer_get_time()-start_sleep_time);
    vTaskDelay(pdMS_TO_TICKS(delay_time / 1000));  // Delay for the remaining time in milliseconds
  }
  else{
    vTaskDelay(1); //keep watchdog happy
  }
}
#endif
#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  // /* Convert from uint8 picture data to int8 */
  // for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; i++) {
  //   input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
  // }
 
#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
  }

  if (kTfLiteOk != GetImage(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  printf("len image data: %d\n", input->bytes);


#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time);
  printf("DC time = %lld\n", dc_total_time);
  printf("conv time = %lld\n", conv_total_time);
  printf("Pooling time = %lld\n", pooling_total_time);
  printf("add time = %lld\n", add_total_time);
  printf("mul time = %lld\n", mul_total_time);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);
  int8_t insect_score = output->data.int8[0];
  printf("%d \n", insect_score);
  
}

