## 创建tensor

 1. 从numpy导入

![numpy导入](https://img-blog.csdnimg.cn/20190710153332121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 2. 从list里面导入

![在这里插入图片描述](https://img-blog.csdnimg.cn/201907101534291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
**tips:** orch.tensor()里面的参数只能是numpy数据或list,即现成的数据；torch.Tensor/FloatTensor()参数为数据的维度,如果是接收现成的数据，要用list形式，如第四行表示方法（不建议使用，易混淆），如果是torch.Tensor/FloatTensor(2,3)表示数据shape为2*3

 1. 创建未初始化数据，即申请内存空间

	三种方法，在数据使用前记得覆盖原有的随机初始化的值，小括号内的参数均表示数据的维度，即shape
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710154728620.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
**Tips:默认类型**
设置默认数据类型的方法：

    torch.set_default_tensor_type(torch.DoubleTensor)

	否则默认数据类型为torch.FloatTensor

 2. 随机初始化

	rand()表示在[0,1]之间均匀采样
	rand_like()的参数是tensor
	randint(min,max,shape)
	randn(shape)表示正态分布的随机数据
	normal(mean,std)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710161307163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 3. 其他常用API
  full(shape, num) 生成一个全为num的，shape为shape的数
 arange/range(min,max,step)
 linspace/logspace(min,max,total_num)
  ones/zeros/eye(shape),其中eye()只能是一维或两维
  randperm(num)随机打散num个数
## 索引与切片

 4. 下标索引
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710162547570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
 5. 切片
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710162846681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
 6. 隔行采样
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710163136926.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
**两个冒号连在一起表示隔行取样：：**，中间没有逗号
 7. 具体维度下的采样
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710163648259.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
 8. **...**任意多的维度
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710163852566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
主要用在取中间的全部，但是最后一个维度另外操作，比如隔行取样
 9. 掩码索引
使用不多，因为会把数据打平
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710164644678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
 10. 打平索引
使用不多
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710164844341.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## 维度变换
操作方法

1. view/reshape()，功能一样
只要保证总的数据维度不变（1.）
忽略位置信息，通道信息，适合全连接层（2.）
只关注宽的信息，把所有照片的行和通道信息合并不关注（3.）
照片和通道信息合并，不关注来自哪张照片和哪个通道，只看行列信息（4.）
恢复数据时必须满足之前的维度信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710171406429.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 12. squeeze()/unsqueeze()
 **unsqueeze()**
 index为正数，则在index之前插入，index是负数，则在index位置之后插入。前后按照正常的从左到右，右边是后面，**插入的是是维度**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710172412808.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710173311599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
**squeeze()**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710175654824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 13. expand/repeat()
**expend()**只能将原来是1的地方变为N，-1表示保持不变
不会主动复制数据，推荐
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710192301568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 14. **repeat()**参数表示对应位置拷贝的次数

        不推荐，因为会更改memory，占有内存变多，不能使用原来的数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710192535470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

 13. ** .t()**

   		矩阵的转置，只能适用于二维矩阵

 14. **Transpose()** 一次只能两两交换![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710194139728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
 15. **permute()**参数对应依次表示变换之前的下标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710194744165.png)


