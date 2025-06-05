"""都开始弄rag知识问答了，那chat/embedding/rerank模型从哪儿弄呢？恩，通过本文你就可以本地部署实现模型部署自由了"""



## inference仓库介绍


| git仓库 | 地址 | 主要功能 | star/fork数
|:-|:-|:-|:-
| xinference |  [xorbitsai/inference](https://github.com/xorbitsai/inference.git) | 模型部署好助手 | 8k/679


#### 功能说明

- 项目是干啥的
  - Xinference是一个性能强大且功能全面的分布式推理框架。可用于大语言模型（LLM），语音识别模型，多模态模型等各种模型的推理。
  - 详细可查看 [xinference 指南](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html)



#### 准备工作

- 1. 接入Xinference本地大模型
    - chat模型
    - embedding模型
    - **支持集群部署 & 通过supervisor进行管理，模型支持上下线**， 还是很方便的


```bash
# step1) 安装环境
# pip install "xinference[vllm]" # xinference[all]
docker pull xprobe/xinference:v1.6.0
docker run \
  -v /tmp/xinference/.xinference:/root/.xinference \
  -v </your/home/path>/.cache/huggingface:/root/.cache/huggingface \
  -v </your/home/path>/.cache/modelscope:/root/.cache/modelscope \
  -p 9997:9997 \
  --gpus all \
  xprobe/xinference:v<your_version> \
  xinference-local -H 0.0.0.0

xinference --help
xinference list # 列出所有模型
# step2) 启用模型
XINFERENCE_HOME=/tmp/xinference xinference-local --host 0.0.0.0 --port 9997
xinference launch --model-engine http://0.0.0.0:9997 --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format gptq --quantization Int4 --gpu_memory_utilization 0.3
# step3) 配置ragflow
xinference engine -e http://0.0.0.0:9997 --model-name qwen2.5-instruct --model-engine vllm -f gptq # 查看模型依赖参数
http://<your-xinference-endpoint-domain>:9997/v1/rerank
http://<your-xinference-endpoint-domain>:9997/v1
# step4) 停止模型
xinference terminate --model-uid "qwen25"
```


![xinference_front](./pics/xinference_front.png)


| 框架名称         | 推荐优先级 | 核心特点                                                                 | 适用场景                                                                 | 使用建议                                                                 |
|------------------|------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **vLLM**         | 优先使用   | - 高吞吐连续批处理<br>- PagedAttention 显存优化<br>- 支持主流模型架构    | 生产环境部署（高并发请求）<br>需要低延迟/高吞吐的在线服务                 | 推荐场景：需要极致性能的 GPU 环境<br>注意：暂不支持非主流架构模型         |
| **SGLang**       | 优先使用   | - 复杂提示结构优化<br>- 嵌套推理加速<br>- 支持 RadixAttention 缓存复用   | 复杂提示工程（如思维链推理）<br>嵌套函数调用/多步骤交互场景               | 推荐场景：提示结构复杂的长文本生成<br>注意：需熟悉其 DSL 语法             |
| **llama.cpp**    | 其次考虑   | - 全量化支持（2-8bit）<br>- CPU/边缘设备优化<br>- 内存占用低             | 资源受限环境（如边缘设备）<br>本地 CPU 推理<br>需要超低精度量化的场景     | 推荐场景：MacBook 本地部署/树莓派等<br>注意：GPU 加速能力有限             |
| **Transformers** | 最后备选   | - 全模型架构支持<br>- 灵活定制推理逻辑<br>- 丰富的前后处理工具           | 快速实验验证<br>非主流模型架构支持<br>需要自定义推理流程的研发阶段        | 推荐场景：原型开发/学术研究<br>注意：原生性能较低，建议搭配优化技术使用   |



## 环境说明 


- 1. 环境说明

```sh
ubuntu1~20.04.2
NVIDIA Quadro 8000
CUDA Version: 12.5
Driver Version: 535.183.01
docker: 20.10.14
Python 3.10.14
```

- 3. python环境
    - python库版本，请看requirements_env.txt

```sh
nltk==3.9.1
vllm==0.8.5.post1
xinference==1.6.0.post1
bitblas>=0.1.0
```