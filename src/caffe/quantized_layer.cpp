/*
 * base_quantization_layer.hpp
 *
 *  Created on: Oct 12, 2016
 *      Author: a0393608
 */

#include "caffe/quantized_layer.hpp"

namespace caffe {


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Quantize_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  if (this->layer_param_.has_quantization_param()) {
    //LOG(INFO) << "Quantizing layer: " << this->layer_param_.name();
    const vector<shared_ptr<Blob > >& blobs = this->blobs();
    const QuantizationParameter& param = this->layer_param_.quantization_param();
    if (param.precision() != QuantizationParameter_Precision_FLOAT) {
      // Trim layer input
      for (int i = 0; i < std::min<int>(param.qparam_in_size(),bottom.size()); ++i) {
        if(param.qparam_in(i).quantize()) {
          this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data<Ftype>(), i, bottom[i]->count());
        }
      }

      // Trim weights - do it only at the start of quantization
      if(param.qparam_w().quantize() && blobs.size() > 0 && param.quantized_infer_count() == 0) {
        this->QuantizeWeights_cpu(blobs[0]->mutable_cpu_data<Ftype>(), blobs[0]->mutable_cpu_connectivity<Ftype>(), blobs[0]->count(), true);
        //if (blobs.size() > 1) { //if (this->bias_term_) {
        //  this->QuantizeWeights_cpu(blobs[1]->mutable_cpu_data<Ftype>(), blobs[1]->count(), false);
        //}
      }

      // Trim layer output
      if(param.qparam_out().quantize()) {
        for (int i = 0; i < top.size(); ++i) {
          this->QuantizeLayerOutputs_cpu(top[i]->mutable_cpu_data<Ftype>(), top[i]->count());
        }
      }
    }
  }
}


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeWeights_cpu(Ftype* data, Ftype* connectivity, const int count, bool clip) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_w = param.qparam_w();
  switch (param.precision()) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_cpu(data, count, param.power2_range(), qparam_w.bitwidth(),
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
void QuantizedLayer<Ftype, Btype>::QuantizeLayerInputs_cpu(Ftype* data, const int blob_id,
      const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_in = param.qparam_in(blob_id);
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, param.power2_range(), qparam_in.bitwidth(),
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
void QuantizedLayer<Ftype, Btype>::QuantizeLayerOutputs_cpu(
      Ftype* data, const int count) {
  const QuantizationParameter& param =  this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_out = param.qparam_out();
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, param.power2_range(), qparam_out.bitwidth(),
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

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2FixedPoint_cpu(Ftype* data, const int cnt, bool power2_range, const int bitwidth,
    const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip) {
  float inv_scale = 1.0f/scale;

  int qrange = unsigned_quant? bitwidth :  (bitwidth - 1);
  Ftype max_data = +(powf(2, qrange) - 1);
  Ftype min_data = unsigned_quant? 0 : -(powf(2, qrange));

  for (int index = 0; index < cnt; ++index) {
    data[index] = (data[index] * scale) + offset;

    // Saturate data
    if(clip) {
      data[index] = std::max(std::min(data[index], max_data), min_data);
    }

    // Round data
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }

    data[index] = (data[index] - offset) * inv_scale;
  }
}

//add by ingenic

template <typename Ftype>
double weightCluster_zero( Ftype weight, int M, Ftype connect, bool clip)
{
  // Saturate data
  if(clip) {
    weight = std::max(std::min((double)weight, pow(2,M)), -pow(2,M));
  }
  double min=100;
  double ind=0;
  double flag=1.0;
  if(connect==0){
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
    return flag*pow(2,ind);
  }else{
    return weight;
  }

}
template double weightCluster_zero<float>(float weight, int M, float connect, bool clip);
template double weightCluster_zero<double>(double weight, int M, double connect, bool clip);
template double weightCluster_zero<unsigned int>(unsigned int weight, int M, unsigned int connect, bool clip);
template double weightCluster_zero<int>(int weight, int M, int connect, bool clip);
  
template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2INQ_cpu(Ftype* data, Ftype* connectivity, const int cnt, bool power2_range,
				      const int bitwidth, const int rounding, int fracbits, float scale, float offset, bool unsigned_quant, bool clip, const float min, const float max) {
  float max_val_abs = std::max(std::fabs(max), std::fabs(min));
  
  //caculate the n1
  int n1=(int)floor(log2(max*4.0/3.0));
  for (int i = 0; i < (cnt); ++i) {
    data[i] = weightCluster_zero(data[i],n1,connectivity[i],clip);
  }
}

//~add by ingenic

template<typename Ftype, typename Btype>
double QuantizedLayer<Ftype, Btype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template void QuantizedLayer<double, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

#ifndef CPU_ONLY
template void QuantizedLayer<double, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float16, double>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float16>::Quantize_cpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
#endif

}

