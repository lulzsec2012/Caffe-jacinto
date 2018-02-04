#include "caffe/quantized_layer.hpp"
#include "caffe/quantized_layer.cuh"
#include "cuda_runtime.h" //add by ingenic
namespace caffe {


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Quantize_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  if (this->layer_param_.has_quantization_param()) {
    //LOG(INFO) << "Quantizing layer: " << this->layer_param_.name();
    const vector<shared_ptr<Blob > >& blobs = this->blobs();
    const QuantizationParameter& param = this->layer_param_.quantization_param();
    if (param.precision() != QuantizationParameter_Precision_FLOAT) {
      // Trim layer input
      for (int i = 0; i < std::min<int>(param.qparam_in_size(),bottom.size()); ++i) {
        if(param.qparam_in(i).quantize()) {
          this->QuantizeLayerInputs_gpu(bottom[i]->mutable_gpu_data<Ftype>(), i, bottom[i]->count());
        }
      }

      // Trim weights - do it only at the start of quantization
      if(param.qparam_w().quantize() && blobs.size() > 0 && param.quantized_infer_count() % 1000 == 0) { //param.sparsity_step_iter()
	if (this->type() == std::string("Convolution")) { //|| this->type() == std::string("InnerProduct")
 	//LOG(INFO)<<"this->name():"<<this->name()<<"in:"<<blobs[0]->is_current_connectivity_valid();	
	  this->QuantizeWeights_gpu(blobs[0]->mutable_gpu_data<Ftype>(), blobs[0]->mutable_gpu_connectivity<Ftype>(), blobs[0]->count(), true);//connectivity
	//this->QuantizeWeights_gpu(blobs[0]->mutable_gpu_data<Ftype>(), blobs[0]->count(), true);
	}else{
	  this->QuantizeWeights_gpu(blobs[0]->mutable_gpu_data<Ftype>(), blobs[0]->count(), true);
	}
        //if (blobs.size() > 1) { //(this->bias_term_) {
        //  this->QuantizeWeights_gpu(blobs[1]->mutable_gpu_data<Ftype>(), blobs[1]->count(), false);
        //}
      }

      // Trim layer output
      if(param.qparam_out().quantize()) {
        for (int i = 0; i < top.size(); ++i) {
          this->QuantizeLayerOutputs_gpu(top[i]->mutable_gpu_data<Ftype>(), top[i]->count());
        }
      }
    }
  }
}

//add by ingenic
template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeWeights_gpu(Ftype* data, Ftype* connectivity, const int count, bool clip) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_w = param.qparam_w();
  switch (param.precision()) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2INQ_gpu(data, connectivity, count, param.power2_range(), qparam_w.bitwidth(),
		 param.rounding_scheme(), qparam_w.fracbits(), qparam_w.scale(),
		 qparam_w.offset(), qparam_w.unsigned_quant(), clip,  qparam_w.min(), qparam_w.max());
    
    //Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_w.bitwidth(),
    //    param.rounding_scheme(), qparam_w.fracbits(), qparam_w.scale(),
    //    qparam_w.offset(), qparam_w.unsigned_quant(), clip);
    break;
  case QuantizationParameter_Precision_FLOAT:
	  break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
    break;
  }
}
//~add by ingenic

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeWeights_gpu(Ftype* data, const int count, bool clip) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_w = param.qparam_w();
  switch (param.precision()) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_w.bitwidth(),
        param.rounding_scheme(), qparam_w.fracbits(), qparam_w.scale(),
        qparam_w.offset(), qparam_w.unsigned_quant(), clip);
    break;
  case QuantizationParameter_Precision_FLOAT:
	  break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
    break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerInputs_gpu(
    Ftype* data, const int blob_id, const int count) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_in = param.qparam_in(blob_id);
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_in.bitwidth(),
          param.rounding_scheme(), qparam_in.fracbits(), qparam_in.scale(),
          qparam_in.offset(), qparam_in.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerOutputs_gpu(Ftype* data,
      const int count) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_out = param.qparam_out();
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_out.bitwidth(),
          param.rounding_scheme(), qparam_out.fracbits(), qparam_out.scale(),
          qparam_out.offset(), qparam_out.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template <typename Dtype>
__global__ void Trim2FixedPoint_kernel(Dtype* data, const int cnt,
    const int bitwidth, const int rounding, float scale, float inv_scale, float offset, float min_data, float max_data, bool clip) {
    CUDA_KERNEL_LOOP(index, cnt) {

    data[index] = (data[index] * scale) + offset;

    // Round data
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = rint(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = __float2int_rd(data[index] + RandUniform_device(index));
      break;
    default:
      break;
    }

    // Saturate data
    if(clip) {
      data[index] = (data[index]>(Dtype)max_data? (Dtype)max_data:
        (data[index]<(Dtype)min_data?(Dtype)min_data:data[index]));
    }

    data[index] = (data[index] - offset) * inv_scale;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2FixedPoint_gpu(Ftype* data, const int cnt, bool power2_range,
      const int bitwidth, const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip) {
  float inv_scale = 1.0f/scale;

  int qrange = unsigned_quant? bitwidth :  (bitwidth - 1);
  float min_data = unsigned_quant? 0 : -(powf(2, qrange));
  float max_data = +(powf(2, qrange) - 1);

  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bitwidth, rounding, scale, inv_scale, offset, min_data, max_data, clip);
}

//add by ingenic
template <typename Dtype>
__global__ void Trim2FixedPoint_kernel_KL(Dtype* data, Dtype* data_tmp, const int cnt,
    const int bitwidth, const int rounding, float scale, float inv_scale, float offset, float min_data, float max_data, bool clip, float * dev_KL_loss) {

    CUDA_KERNEL_LOOP(index, cnt) {

    data_tmp[index] = (data[index] * scale) + offset;

    // Round data
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data_tmp[index] = rint(data_tmp[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data_tmp[index] = __float2int_rd(data_tmp[index] + RandUniform_device(index));
      break;
    default:
      break;
    }

    // Saturate data
    if(clip) {
      data_tmp[index] = (data_tmp[index]>(Dtype)max_data? (Dtype)max_data:
        (data_tmp[index]<(Dtype)min_data?(Dtype)min_data:data_tmp[index]));
    }
    if(data_tmp[index]>-0.00001 && data_tmp[index] < 0.00001){
      data_tmp[index] = 0;
    }else{
    data_tmp[index] = (data_tmp[index] - offset) * inv_scale;
    data_tmp[index] = data[index] * log(data[index]/data_tmp[index]);
    }
    //data_tmp[index] = (data_tmp[index] - offset) * inv_scale - data[index];
    //data_tmp[index] = abs(data_tmp[index]);
  }
  
  for(int i = 1;i < blockDim.x ; i <<= 1){
    if(threadIdx.x % (i<<1) == i){
        data_tmp[threadIdx.x - i] += data_tmp[threadIdx.x];
    }
    __syncthreads();        
  }    

  if(threadIdx.x == 0)    *dev_KL_loss = data_tmp[0];
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2FixedPoint_gpu_KL(Ftype* data, const int cnt, bool power2_range,
      const int bitwidth, const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip) {
  float inv_scale = 1.0f/scale;

  int qrange = unsigned_quant? bitwidth :  (bitwidth - 1);
  float min_data = unsigned_quant? 0 : -(powf(2, qrange));
  float max_data = +(powf(2, qrange) - 1);
  //
  Ftype *data_tmp = 0;  
  cudaError_t cudaStatus = cudaMalloc((void**)&data_tmp, cnt * sizeof(Ftype));  
  if (cudaStatus != cudaSuccess) {  
     fprintf(stderr, "cudaMalloc failed!");
     LOG(FATAL) << "cudaMalloc failed!" << " for layer:" << this->layer_param_.name();
     cudaFree(data_tmp);  
  }
  float *dev_KL_loss = 0;
  cudaStatus = cudaMalloc((void**)&dev_KL_loss, 1 * sizeof(float));  
  if (cudaStatus != cudaSuccess) {  
     LOG(INFO)<<"cudaMalloc failed!";
     cudaFree(dev_KL_loss);
     dev_KL_loss = 0;
  }
  float min_factor=0.5;
  float max_factor=1.2;
  int   step = 100;
  float factor_step = (max_factor - min_factor)/step;
  float best_KL_loss = 10000;
  int best_step = 71;//71
  for(int i=0;i<step;i++){
     float KL_loss = 10000;
     float scale_cur = scale * (min_factor + factor_step*i);
     inv_scale = 1.0f/scale_cur;
     
     Trim2FixedPoint_kernel_KL<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(	
      data, data_tmp, cnt, bitwidth, rounding, scale_cur, inv_scale, offset, min_data, max_data, clip, dev_KL_loss);
     // Copy output vector from GPU buffer to host memory.  
     cudaStatus = cudaMemcpy(&KL_loss, dev_KL_loss, 1 * sizeof(float), cudaMemcpyDeviceToHost);  
     if (cudaStatus != cudaSuccess) {
       LOG(INFO)<<"Copy output vector failed!";
       cudaFree(dev_KL_loss);
       dev_KL_loss = 0;
     }
     if(0==i){
       best_KL_loss = KL_loss;
     }
     if(best_KL_loss > KL_loss){
       best_KL_loss = KL_loss;
       best_step = i;
       //LOG(INFO)<<"hello inegnic!"<<i<<" KL_loss="<<KL_loss;
     }
  }
  if(data_tmp != NULL){
    cudaFree(data_tmp);
    data_tmp = NULL;
  }
  if (dev_KL_loss != NULL) {
     cudaFree(dev_KL_loss);
     dev_KL_loss = 0;
  }

  LOG(INFO) << "Best KL factor is:" << min_factor + factor_step*best_step <<" best_step=" << best_step << " for layer:" << this->layer_param_.name();
  scale = scale * (min_factor + factor_step*best_step);
  //scale = scale * 1.0f;	
  inv_scale = 1.0f/scale;
  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bitwidth, rounding, scale, inv_scale, offset, min_data, max_data, clip);
}

//~add by ingenic

//add by ingenic
template <typename Dtype>
  __global__ void Trim2FixedPointINQ_kernel(Dtype* data, Dtype* connectivity, const int cnt,
    const int bitwidth, const int rounding, float scale, float inv_scale, float offset, float min_data, float max_data, bool clip) {
    CUDA_KERNEL_LOOP(index, cnt) {
      if(connectivity[index] < 0.00001 && connectivity[index] > -0.00001){
	data[index] = (data[index] * scale) + offset;
	
	// Round data
	switch (rounding) {
	case QuantizationParameter_Rounding_NEAREST:
	  data[index] = rint(data[index]);
	  break;
	case QuantizationParameter_Rounding_STOCHASTIC:
	  data[index] = __float2int_rd(data[index] + RandUniform_device(index));
	  break;
	default:
	  break;
	}
	
	// Saturate data
	if(clip) {
	  data[index] = (data[index]>(Dtype)max_data? (Dtype)max_data:
			 (data[index]<(Dtype)min_data?(Dtype)min_data:data[index]));
	}
	
	data[index] = (data[index] - offset) * inv_scale;
      }
    }
 }
 
template <typename Dtype>
__global__ void Trim2INQ_kernel(Dtype* data, const int M,
    Dtype* connectivity, bool clip, const int cnt) {
  CUDA_KERNEL_LOOP(index, cnt) {
    
    Dtype weight;
    // Saturate data
    if(true) {
      weight = max(min(data[index], pow(2,M)), -pow(2,M));
    }else{
      weight = data[index];
    }     
    double min=100;
    double ind=0;
    double flag=1.0;
    if(connectivity[index] < 0.00001 && connectivity[index] > -0.00001){
      if(min>std::abs(weight))
	{
	  min=std::abs(weight);
	  flag=0.0;
	}
      
      for(int i=(M-6);i<=M;i++)
	{
	  if(min>std::abs(weight-pow(2,i)))
	    {
	      min=std::abs(weight-pow(2,i));
	      ind=i;
	      flag=1.0;
	    }
	  if(min>std::abs(weight+pow(2,i)))
	    {
	      min=std::abs(weight+pow(2,i));
	      ind=i;
	      flag=-1.0;
	    }
	}
      data[index] = flag*pow(2,ind);
    }else{
      data[index] = weight;
    } 
  }
}
  
template<typename Ftype, typename Btype>
  void QuantizedLayer<Ftype, Btype>::Trim2INQ_gpu(Ftype* data, Ftype* connectivity, const int cnt, bool power2_range,
				      const int bitwidth, const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip, const float min, const float max) {
  
  float max_val_abs = std::max(std::fabs(max), std::fabs(min));
  //caculate the n1
  int n1=(int)floor(log2(max*4.0/3.0));
  Trim2INQ_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(data,n1,connectivity,clip,cnt);

  //
  float inv_scale = 1.0f/scale;

  int qrange = unsigned_quant? bitwidth :  (bitwidth - 1);
  float min_data = unsigned_quant? 0 : -(powf(2, qrange));
  float max_data = +(powf(2, qrange) - 1);

  //Trim2FixedPointINQ_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
  //     data, connectivity, cnt, bitwidth, rounding, scale, inv_scale, offset, min_data, max_data, clip);
  //
}

//~add by ingenic

template void QuantizedLayer<double, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float16, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);


}  // namespace caffe


