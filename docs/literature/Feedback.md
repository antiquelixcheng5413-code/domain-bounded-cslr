### Feedback for the first one：



郝同学的提问是我们做文献综述时最需要先弄清楚的部分，思路是很好的。

#####

#### Answer for question:



##### **项目类型：**

我们这个项目目前不是静态手势识别，也不是完整的连续手语翻译，而是更接近 **domain-bounded Chinese Sign Language recognition with semantic reconstruction**。也就是说，核心识别任务是中国手语 CSL 下的有限场景词汇和短语识别，应用场景会控制在医院前台、校园服务或公共服务问询这类固定范围内。具体需要参考数据来源以及自采数据，如果后续自采数据不足，我们会把系统明确写成 controlled technical prototype。



##### **项目目标：**

输入是摄像头或视频中的手语动作，系统先提取 hands、upper body、face 的 **landmarks**，把每一帧的关键点转成一个向量，再把连续帧组成 **time-series / temporal-sequence data**。使用的 time series不是传统统计预测里的 ARIMA 序列，而是计算机视觉中的时序动作序列建模。之后系统会用 LSTM、BiLSTM、TCN 或 compact Transformer 学习动作随时间变化的规律，输出识别到的 gloss / intent，并进一步转成可读的中文表达。

#####

##### **项目边界：**

**Public dataset + individual getting table.** 计划一个有限词汇、有限短语、有限场景的 CSL 识别与语义重构原型。词汇表初期会控制在 30–50 个领域相关 signs，短语大概 10–20 个，后续根据数据采集质量再调整。我们会重点评价 Top-1 accuracy、macro-F1、confusion matrix、signer-aware accuracy、latency、FPS 和资源占用，而不是单一 random split accuracy。



##### **明确不做的部分（ future work）：**

连续手语识别、完整翻译、复杂双流大模型。目前初期选择一个更可控、更可评价的范围。



##### **特征提取：**

你提到 RGB、CNN、3D CNN、MediaPipe Hands 等路线。

&#x09;这里我们现在的选择是：

&#x09;以 MediaPipe / OpenPose-style landmarks 作为 baseline，优先使用 holistic landmarks (hands + upper body + face)。原因是 CSL 不只是手型，面部表情、身体位置、动作轨迹都可能影响意义。

&#x09;RGB hand / face crops 不会一开始就作为主线，而是作为可选增强。（当 error analysis 发现 landmark-only 模型在某些相似 signs 上确无法分辨时）



##### **文献的分类：**

对于语言学基础、静态手势识别、孤立词识别、连续识别、翻译和应用系统分开管理的思维很好，这一点和我们现在综述的结构相对应。



Overall，问题很准确，感谢你的分享。



### Feedback for the second one：



王同学整体资料整理得比较完整，覆盖了项目边界、文献分类、技术路线、实验设计、评价指标和术语表，对 SLR、ISLR、CSLR、SLT 这些层级已经有了比较系统的认识。

提到 random split 和 signer-independent split 的区别，以及 FPS、latency、model size 这些实时系统评价指标，这些和现在项目的评价方向是比较一致，总体方向思考很完整充足。



improve part：文献分析部分可以加强不同论文之间的比较和联系。

实时孤立词识别是重要考量，但直接转化标签或许不是面向对象的一个系统，可以考虑增加 semantic reconstruction，把识别出的 gloss / intent 进一步转成可读中文。



overall，这份整理的基础是比较扎实的，信息量也很足，感谢你的分享。
