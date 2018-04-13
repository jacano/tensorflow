#include "tfliteextern.h"

tflite::FlatBufferModel* tfeFlatBufferModelBuildFromFile(char* filename)
{
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename); 
  return model.release();
}

tflite::FlatBufferModel* tfeFlatBufferModelBuildFromBuffer(char* buffer, int bufferSize)
{
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(buffer, bufferSize); 
  return model.release();
}

/*
tflite::FlatBufferModel* tfeFlatBufferModelBuildFromModel(tflite::FlatBufferModel* other)
{
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromModel(other);
  return model.release();
}
*/

void tfeFlatBufferModelRelease(tflite::FlatBufferModel** model)
{
  delete *model;
  *model = 0;
}

tflite::ops::builtin::BuiltinOpResolver* tfeBuiltinOpResolverCreate(tflite::OpResolver** opResolver)
{
   tflite::ops::builtin::BuiltinOpResolver* builtinOpResolver = new tflite::ops::builtin::BuiltinOpResolver();
   *opResolver = dynamic_cast<tflite::OpResolver*>(builtinOpResolver);
   return builtinOpResolver;
}
void tfeBuiltinOpResolverRelease(tflite::ops::builtin::BuiltinOpResolver** resolver)
{
  delete * resolver;
  *resolver = 0;
}
tflite::Interpreter* tfeInterpreterCreate()
{
  return new tflite::Interpreter();
}
int tfeInterpreterAllocateTensors(tflite::Interpreter* interpreter)
{
  return interpreter->AllocateTensors();
}
int tfeInterpreterInvoke(tflite::Interpreter* interpreter)
{
  return interpreter->Invoke();
}
char* tfeInterpreterInputTensor(tflite::Interpreter* interpreter, int index)
{
  return interpreter->typed_input_tensor<char>(index);
}
char* tfeInterpreterOutputTensor(tflite::Interpreter* interpreter, int index)
{
  return interpreter->typed_output_tensor<char>(index);
}
int tfeInterpreterTensorSize(tflite::Interpreter* interpreter)
{
  return interpreter->tensors_size();
}
int tfeInterpreterNodesSize(tflite::Interpreter* interpreter)
{
  return interpreter->nodes_size();
}
int tfeInterpreterGetInputSize(tflite::Interpreter* interpreter)
{
  return interpreter->inputs().size();
}
void tfeInterpreterGetInput(tflite::Interpreter* interpreter, int* input)
{
  std::vector<int> ivec = interpreter->inputs();
  memcpy(input, &ivec[0], ivec.size());
}
const char* tfeInterpreterGetInputName(tflite::Interpreter* interpreter, int index)
{
  return interpreter->GetInputName(index);
}
int tfeInterpreterGetOutputSize(tflite::Interpreter* interpreter)
{
  return interpreter->outputs().size();
}
void tfeInterpreterGetOutput(tflite::Interpreter* interpreter, int* output)
{
  std::vector<int> ovec = interpreter->outputs();
  memcpy(output, &ovec[0], ovec.size());
}
const char* tfeInterpreterGetOutputName(tflite::Interpreter* interpreter, int index)
{
  return interpreter->GetOutputName(index);
}

void tfeInterpreterRelease(tflite::Interpreter** interpreter)
{
  delete * interpreter;
  *interpreter = 0;
}

tflite::InterpreterBuilder* tfeInterpreterBuilderCreate(tflite::FlatBufferModel* model, tflite::OpResolver* opResolver)
{
  return new tflite::InterpreterBuilder(*model, *opResolver);
}
void tfeInterpreterBuilderRelease(tflite::InterpreterBuilder** builder)
{
  delete * builder;
  *builder = 0;
}
void tfeInterpreterBuilderBuild(tflite::InterpreterBuilder* builder, tflite::Interpreter* interpreter)
{
  std::unique_ptr<tflite::Interpreter> ptr(interpreter);
  (*builder)(&ptr);
  ptr.release();
}
