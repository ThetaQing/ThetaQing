## where

    torch.where(condition,x,y) #condition必须是tensor类型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190712194225725.png)
condition的维度和x，y一致，用1和0分别表示该位置的取值
例：输入：

    cond = torch.tensor([[0.6, 0.7],
                        [0.3, 0.6]])
    a = torch.tensor([[1., 1.],
                     [1., 1.]])
    b = torch.tensor([[0., 0.],
                     [0., 0.]])
    c = torch.where(cond > 0.5, a, b) #此时cond只有0和1的值
    print(c)
输出：

    tensor([[1., 1.],
            [0., 1.]])

高度并行

## gather

    torch.gather(input, dim, index, out=None)
相当于查表操作
举例：

    prob = torch.randn(4, 10)
    idx = prob.topk(dim=1, k=3)  # prob在维度1中前三个最大的数，一共有4行，返回值和对应的下标
    print("all of topk idx: ", idx)
    idx = idx[1]
    print("idx[1]: ", idx)
    label = torch.arange(10) + 100  # 举个例子，这里的列表表示为
    # 0对应于100,1对应于101，以此类推，根据实际应用修改
    result = torch.gather(label.expand(4, 10), dim=1, index=idx.long())  # lable相当于one-hot编码，index表示索引
    # 换而言是是y与x的函数映射关系，index表示x
    print("result:", result)

输出结果为：

    all of topk idx:  torch.return_types.topk(
    values=tensor([[0.7878, 0.2928, 0.2062],
            [0.2524, 0.2094, 0.0350],
            [1.5519, 0.8405, 0.7521],
            [1.3380, 0.9290, 0.5655]]),
    indices=tensor([[2, 0, 8],
            [9, 5, 6],
            [1, 2, 0],
            [3, 7, 8]]))
    idx[1]:  tensor([[2, 0, 8],
            [9, 5, 6],
            [1, 2, 0],
            [3, 7, 8]])
    result: tensor([[102, 100, 108],
            [109, 105, 106],
            [101, 102, 100],
            [103, 107, 108]])
