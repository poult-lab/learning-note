# Complete collection of functions and commands

## Common commands conda & pip (python)

#### 1. View the current Python version

```bash
python --version
```

#### 2. view conda environment

```bash
conda info --env
```

#### 3. enter environment

```bash
conda activate XXX
```

#### 4. Create a virtual environment with specified [ˈspesɪfaɪd]  python for your project 

```bash
conda create -n yourenvname python=x.x anaconda
```

#### 5. Delete a no longer needed virtual environment

```bash
conda remove -n yourenvname -all
```

#### 6. Deactivate your virtual environment.

```bash
source deactivate
```

#### 7. Check conda is installed and in your PATH

```bash
$ conda -V
conda 3.7.0
```

#### 8. Check conda is up to date

```bash
conda update conda
```

####  9. Check conda list

```bash
conda list
```

#### 10. install /uninstall a package

```bash
conda install package
```

#### 11.pip install a specific version, type the package name followed by the required version:

```bash
pip install 'PackageName==1.4'
```



## Slurm Workload Manager

The **Slurm Workload Manager**, formerly known as **Simple Linux Utility for Resource Management** (**SLURM**), or simply **Slurm**, is a [free and open-source](https://en.wikipedia.org/wiki/Free_and_open-source) [job scheduler](https://en.wikipedia.org/wiki/Job_scheduler) for [Linux](https://en.wikipedia.org/wiki/Linux) and [Unix-like](https://en.wikipedia.org/wiki/Unix-like) [kernels](https://en.wikipedia.org/wiki/Kernel_(operating_system)), used by many of the world's [supercomputers](https://en.wikipedia.org/wiki/Supercomputer) and [computer clusters](https://en.wikipedia.org/wiki/Computer_cluster).









## Common commands linux

#### 1.How do I run .sh file shell script in Linux?

1. Open the Terminal application on Linux or Unix

2. Create a new script file with .sh extension using a text editor

3. Write the script file using nano script-name-here.sh

4. Set execute permission on your script using chmod command :

   ```bash
   chmod +x script-name-here.sh
   ```

5. To run your script :

   ```bash
   ./script-name-here.sh
   ```

   Another option is as follows to execute shell script:

   ```bash
   sh script-name-here.sh
   ```

   OR

   ```bash
   bash script-name-here.sh
   ```

   

#### 2.Bash set -x/+x to print statements as they are executed

Setting the -x tells Bash to print out the statements as they are being executed. It can be very useful as a logging facility and for debugging when you need to know which statements were execute and in what order.

It can be enabled on the command line or on the sh-bang line by providing -x or by the set -x statement.

It can be disabled using the set +x statement.

See this example:

**examples/shell/set-x.sh**

```bash
#!/bin/bash -x 

name="Foo"
echo $name 

set +x 

age=42
echo $age  

set -x 

language=Bash
echo $language
```



and the output it generates:

```shell
$ ./examples/shell/set-x.sh

+ name=Foo
+ echo Foo
Foo
+ set +x
42
+ language=Bash
+ echo Bash
Bash
```



#### 3.export variable

`export` propagates the variable to subprocesses.

For example, if you did

```bsh
FOO=bar
```

then a subprocess that checked for FOO wouldn't find the variable whereas

```bsh
export FOO=bar
```

would allow the subprocess to find it.

But if `FOO` has *already* been defined as an environment variable, then `FOO=bar` will modify the value of that environment variable.

For example:

```bsh
FOO=one     # Not an environment variable
export FOO  # Now FOO is an environment variable
FOO=two     # Update the environment variable, so sub processes will see $FOO = "two"
```

Older shells didn't support the `export FOO=bar` syntax; you had to write `FOO=bar; export FOO`.

------

Actually, if you don't use "`export`", you're not defining an environment variable, but just a shell variable. Shell variables are only available to the shell process; environment variables are available to *any* subsequent process, not just shells. In addition, subshells are commands contained within parentheses, which do have access to shell variables, whereas what you're talking about are child processes that happen to be shells.



#### 4.Brace parameter expansion

Let’s run a couple of examples to understand how parameter expansion works:

```bash
$ price=5
$ echo "$priceUSD"

$ echo "${price}USD"
5USD
```

As we can observe from the output, **we use \*{..}\* here as a disambiguation mechanism for delimiting tokens**.

Adding double quotes around a variable tells Bash to treat it as a single word. Hence, the first *echo* statement returns an empty output since the variable *priceUSD* is not defined.

However, **using \*${price}\* we were able to resolve the ambiguity**. Clearly, in the second *echo* statement, we could concatenate the string *USD* with the value of the *price* variable that was what we wanted to do.



#### 5.*if [ ! -d $directory ]*

In bash command -d is to check if the directory exists or not.

For example, I having a directory called 

> /home/sureshkumar/test/.

The directory variable contains:

> "/home/sureshkumar/test/"
>
> if [ -d $directory ]

This condition will be true only when a directory exists. 

I have changed the directory variable to

>  "/home/a/b/". 

This directory does not exist.

> if [ -d $directory ]



Now, the condition is false. If I put the ! in front of my directory does not exist, then this if the condition is true. If this directory does exists then the 

> if [ ! -d $directory ] 
>
> condition is false.

The procedure of the ! the operator is if this condition is true, then it gives the condition is false. If the command is false then it says the command is true. This is the work of! operator.

> if [ ! -d $directory ]

This command true only if the $directory does not exist. If the directory exists, it returns false.



#### 6.mkdir -p

With the help of mkdir -p command you can create sub-directories of a directory. It will create parent directory first, if it doesn't exist. But if it already exists, then it will not print an error message and will move further to create sub-directories.

```bash
mkdir -p $base_exp_dir
```



#### 7.bash 返回上一级命令

```bash
cd ..
```



#### 8.real-time monitor GPU(ubuntu)

```bash
watch nvidia-smi
```





## Miscellaneous options [ˌmɪsəˈleɪniəs ˈɒpʃnz] 

```python
python -W ignore XXX.py
```

Warning control. Python’s warning machinery by default prints warning messages to [`sys.stderr`](https://docs.python.org/3/library/sys.html#sys.stderr).

The simplest settings apply a particular action unconditionally to all warnings emitted by a process (even those that are otherwise ignored by default):

```
-Wdefault  # Warn once per call location
-Werror    # Convert to exceptions
-Walways   # Warn every time
-Wmodule   # Warn once per calling module
-Wonce     # Warn once per Python process
-Wignore   # Never warn
```



## About Pytorch and torchvision

**tuple**:A tuple looks just like a list except you use parentheses instead of square brackets. Once you define a tuple, you can access individual elements by using each item’s index, just as you would for a list. 

For example, if we have a rectangle that should always be a certain size, we can ensure that its size doesn’t change by putting the dimensions into a tuple:

for example:

```python
dimensions = (200, 50)

print(dimensions[0])

print(dimensions[1])
```

keyword: can't be changed   ()



**list**: In Python, square brackets ([]) indicate a list, and individual elements in the list are separated by commas. Here’s a simple example of a list that contains a few kinds of bicycles:

for example:

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized'] 

print(bicycles)
```

keyword: can be changed  orderly []

**hint**: 当索引列表时，索引值为negative. (-1永远表示最后一项)

```python
li=[1,2,3,4,5,6]

print("This is test: ",li[-1])

print("This is test: ",li[-2])

print("This is test: ",li[-3])
```

output:

```powershell
This is test:  6 

This is test:  5 

This is test:  4
```



**dictionary**: Consider a game featuring aliens that can have different colors and point values. This simple dictionary stores information about a particular alien

字典是另一种可变容器模型，且可存储任意类型对象。

字典的每个键值 **key=>value** 对用冒号 : 分割，每个键值对之间用逗号 , 分割，整个字典包括在花括号 {} 中 ,格式如下所示：

for example:

```python
alien_0 = {'color': 'green', 'points': 5} 

print(alien_0['color']) 

print(alien_0['points'])
```

keyword: {} is fast of inserting and searching  waste memory



#### dict.keys()

```python
tinydict = {'Name': 'Zara', 'Age': 7}

print("Value : %s" %  tinydict.keys())
```

以列表返回一个字典所有的键

```powershell
Value : ['Age', 'Name']
```



**nested dictionary**:

```python
#多级字典（嵌套字典）
FamousDict ``=` `{
 ``'薛之谦'``:{
  ``'身高'``:``178``,
  ``'体重'``:``130``,
  ``'口头禅'``:[``'你神经病啊！'``,``'我不要面子啊'``] ``#相应的值可以是 一个列表
 ``},
 ``'吴青峰'``:{
  ``'身高'``:``170``,
  ``'体重'``:``120``,
  ``'口头禅'``:[``'我叫吴青峰'``,``'你好'``]
 ``}
}
#访问多级字典：
print``(``'薛之谦的体重为：'``,FamousDict[``'薛之谦'``][``'体重'``],``'斤'``)
#修改薛之谦体重为125
FamousDict[``'薛之谦'``][``'体重'``] ``=` `125
print``(``'减肥后的薛之谦体重为：'``,FamousDict[``'薛之谦'``][``'体重'``],``'斤'``)
#新添薛之谦腰围100
FamousDict[``'薛之谦'``][``'腰围'``] ``=` `100
print``(``'薛之谦的腰围为：'``,FamousDict[``'薛之谦'``][``'腰围'``],``'cm'``)
#多级字典删除
FamousDict[``'吴青峰'``].pop(``'身高'``) ``#标准删除
del` `FamousDict[``'吴青峰'``][``'体重'``] ``#另一个删除方法
print``(``'关于吴青峰现在只剩下：'``,FamousDict[``'吴青峰'``])
```



**set**:

for example:

```python
\>>> basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
\>>> **print**(basket)            # 这里演示的是去重功能
{'orange', 'banana', 'pear', 'apple'}
```

keyword: no orderly  on repeat



**slice from list**:

1. 切片操作不是列表特有的，python中的有序序列都支持切片，如字符串，元组。
2. 切片的返回结果类型和切片对象类型一致，返回的是切片对象的子序列，如：对一个列表切片返回一个列表，
3. 字符串切片返回字符串。
4. 切片生成的子序列元素是源版的拷贝。因此切片是一种浅拷贝。

```python
li=["A","B","C","D"]
```

格式： li[start : end : step]  

start是切片起点索引，end是切片终点索引，但切片结果不包括**终点索引的值**。step是步长默认是1。

```python
t=li[0:3]      ["A","B","C"]     #起点的0索引可以省略，t=li[:3]

t=li[2: ]      ["C","D"]         #省略end，则切到末尾

t=li[1:3]      ["B","C"]

t=li[0:4:2]    ["A","C"]         #从li[0]到li[3],设定步长为2。
```

如何确定start和end，他们是什么关系？

 在step的符号一定的情况下，start和end可以混合使用正向和反向索引，无论怎样，你都要保证start和end之间有和step方向一致元素 间隔，否则会切出空列表.

```python
       t=li[0:2]

​      t=li[0:-2]

​      t=li[-4:-2]

​      t=li[-4:2]

# 上面的结果都是一样的；t为["A","B"]
```

![20150702234502400](/home/jiang/桌面/About Python and some image algorithm/pictures source/20150702234502400.png)

```python
     t=li[-1:-3:-1]

​     t=li[-1:1:-1]

​     t=li[3:1:-1]

​     t=li[3:-3:-1]

# 上面的结果都是一样的；t为["D","C"]
```

![20150702234736704](/home/jiang/桌面/About Python and some image algorithm/pictures source/20150702234736704.png)

```python
      t=li[-1:-3]

​     t=li[-1:1]

​     t=li[3:1]

​     t=li[3:-3]



# 都切出空列表
```

![20150702235107635](/home/jiang/桌面/About Python and some image algorithm/pictures source/20150702235107635.png)

 同时，step的正负决定了切片结果的元素采集的先后,省略**start 和 end**表示以原列表全部为目标.

```python
   t=li[::-1]   t--->["D","C","B","A"]   #反向切，切出全部

   t=li[:]    t--->["A","B","C","D"]     #正向切全部
```



**slice from Numpy**:

```python
import numpy as np  

a = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7]])
print(a[..., 1])  # 第2列元素
print(a[1, ...])  # 第2行元素
print(a[..., 1:])  # 第2列及剩下的所有元素
```

output:

```bash
[2 4 5]
[3 4 5 6]
[[2 3 4]
 [4 5 6]
 [5 6 7]]
```



#### 1. About decorator of python

for example:

```python
def foo():
    print('i am foo')
```

现在有一个新的需求，希望可以记录下函数的执行日志，于是在代码中添加日志代码：

```python
def foo():
    print('i am foo')
	logging.info("foo is running")
```

对bar(), bar()n 有一样的需求

```python
def use_logging(func):
	logging.warn("%s is running" % func.__name__)
	func()

def bar():
    print('i am bar')
    
use_logging(bar)
```

simple example

```python
def use_logging(func):
    
    def wrapper(*args, **kwargs):
        logging.warn("%s is running" % func.__name__)
        return func(*args)
    return wrapper

@use_logging
def foo():
    print("i am foo")
    
@use_logging
def bar():
    print("i am bar")
    
bar()
```



#### 2. About backward() from pytorch 

```python
import torch as t
from torch.autograd import Variable as v

a = v(t.FloatTensor([2, 3]), requires_grad=True)  
print("This is a: ", a)  
b = a + 3
c = b * b * 3
out = c.mean() # not merely transfer to scaler
print("This is out: ", out)
out.backward(retain_graph=True) # 这里可以不带参数，默认值为‘1’，由于下面我们还要求导，故加上retain_graph=True选项

print(a.grad) # tensor([15., 18.])
print("This is b.grad: \n", b.requires_grad)
```



output:

```powershell
This is a:  tensor([2., 3.], requires_grad=True)
This is out:  tensor(91.5000, grad_fn=<MeanBackward0>)
tensor([15., 18.])
This is b.grad: 
 True
```

`注意参数requires_grad=True`让其成为一个叶子节点，具有求导功能。

![1](/home/jiang/桌面/About Python and some image algorithm/pictures source/1.png)

手动求导结果：

![2](/home/jiang/桌面/About Python and some image algorithm/pictures source/2.png)

#### 3. About torch.squeeze() & torch.unsqueeze()

torch.squeeze(): 

这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度(这里面为第N维)。还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。

torch.unsqueeze()

这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度

```python
x = torch.ones(1, 2, 3, 4, 5)
y = torch.squeeze(x)  # 返回一个张量，其中所有大小为1的输入的维都已删除
z = torch.squeeze(
    x, 3
)  # Returns a tensor with third dimensions of input of size 1 removed.
w = torch.unsqueeze(x, 4)  # 在第四个维度增加一个维度，并且增加的这个维度是一
print("This is x: ", x.size())
print("This is y: ", y.size())
print("This is z: ", z.size())
print("This is w: ", w.size())
x = x * 2
print("This is new x: ", x)
print("This is new y: ", y)
q = torch.unsqueeze(x, -1)
print("This is q: ", q.size())
h = torch.unsqueeze(x, -2)
print("This is h: ", h.size())
i = torch.unsqueeze(x, 5)
print("This is i: ", i.size())
```



out:

```powershell
This is x:  torch.Size([1, 2, 3, 4, 5])
This is y:  torch.Size([2, 3, 4, 5])
This is z:  torch.Size([1, 2, 3, 4, 5])
This is w:  torch.Size([1, 2, 3, 4, 1, 5])
This is new x:  tensor([[[[[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]],

          [[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]],

          [[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]]],


         [[[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]],

          [[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]],

          [[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]]]]])
This is new y:  tensor([[[[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]]],


        [[[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]]]])
This is q:  torch.Size([1, 2, 3, 4, 5, 1])
This is h:  torch.Size([1, 2, 3, 4, 1, 5])
This is i:  torch.Size([1, 2, 3, 4, 5, 1])
```



#### 4. About h5py



```python
dir_video = "data/visual_feature.h5"

dir_audio = "data/audio_feature.h5"

dir_labels = "data/labels.h5"

dir_order_test = "data/test_order.h5"

# with h5py.File(dir_video, "r") as hf:

#   video_features = hf["avadataset"][:1]

# print("This is video feature: ", video_features)



# HDF5的读取：

f = h5py.File(dir_video, "r") # 打开h5文件

# 可以查看所有的主键

for key in f.keys():

  print("This is key: ", key)

  # print(f[key].name)

  print("This is shape of dir_video: ", f[key].shape)

  # print(f[key].value)



f = h5py.File(dir_audio, "r") # 打开h5文件

# 可以查看所有的主键

for key in f.keys():

  print("This is key: ", key)

  # print(f[key].name)

  print("This is shape of dir_audio: ", f[key].shape)

  # print(f[key].value)



f = h5py.File(dir_labels, "r") # 打开h5文件

# 可以查看所有的主键

for key in f.keys():

  print("This is key: ", key)

  # print(f[key].name)

  print("This is shape of dir_labels: ", f[key].shape)

  # print(f[key].value)



f = h5py.File(dir_order_test, "r") # 打开h5文件

# 可以查看所有的主键

for key in f.keys():

  print("This is key: ", key)

  # print(f[key].name)

  print("This is shape of dir_order_test: ", f[key].shape)

  # print(f[key].value)
```



#### 5. about copy_ from "tensor"

```python
a = torch.zeros(3)
print("This is a: ", a)
b = torch.randn(3)
print("This is b: ", b)
a.copy_(b) #从输出来看这里面将b的内容复制给a

print("This is a new a: ", a)
print("This is a new b: ", b)
```



```powershell
This is a:  tensor([0., 0., 0.])
This is b:  tensor([-1.6451, -0.0094, -0.4104])
This is a new a:  tensor([-1.6451, -0.0094, -0.4104])
This is a new b:  tensor([-1.6451, -0.0094, -0.4104])
```



#### 6. about variable function from torch



```python
from torch.autograd import Variable
import matplotlib.pyplot as plt

print("构建函数y=x^2,并求x=3导数")
# x = np.arange(-3, 3.01, 0.1)
# # print("This is x: ",x)
# y = x ** 2
# plt.plot(x, y)
# plt.plot(2, 4, "ro")
# plt.show()
x = Variable(torch.FloatTensor([7]), requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```

output: 

```powershell
tensor([14.])


```



#### 7. torch.max(input, dim) 函数

输入

- `input`是softmax函数输出的一个`tensor`
- `dim`是max函数索引的维度`0/1`，`0`是每列的最大值，`1`是每行的最大值

输出

- 函数会返回两个`tensor`，第一个`tensor`是每行的最大值；第二个`tensor`是每行最大值的索引。



tip: 在多分类任务中我们并不需要知道各类别的预测概率，所以返回值的第一个`tensor`对分类任务没有帮助，而第二个`tensor`包含了预测最大概率的索引，所以在实际使用中我们仅获取第二个`tensor`即可。



```python
import torch
a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
print(a)
```

output:

```powershell
tensor([[ 1,  5, 62, 54],
        [ 2,  6,  2,  6],
        [ 2, 65,  2,  6]])
```

索引每行的最大值：

```python
print(torch.max(a, 1))
```

output:

```powershell
torch.return_types.max(
values=tensor([62,  6, 65]),
indices=tensor([2, 3, 1]))
```

在计算准确率时第一个tensor `values`是不需要的，所以我们只需提取第二个tensor，并将tensor格式的数据转换成array格式。

```python
print("This is: ", torch.max(a, 1)[1].numpy())
```

output:

```powershell
This is:  [2 3 1]
```



#### 8.1 about softmax(Tensor,dim) function

```python
import torch


import torch.nn.functional as F


x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


y1 = F.softmax(x, dim=0) # 对每一列进行softmax

print(y1)

print("This is y1[0]:", y1[0])

print("This is y1[1]:", y1[1])



y2 = F.softmax(x, dim=1) # 对每一行进行softmax

print(y2)
print("This is y2[0]:", y2[0])
print("This is y2[1]:", y2[1])
```



This is output:

```powershell
tensor([[3.2932e-04, 3.2932e-04, 3.2932e-04, 3.2932e-04],
        [1.7980e-02, 1.7980e-02, 1.7980e-02, 1.7980e-02],
        [9.8169e-01, 9.8169e-01, 9.8169e-01, 9.8169e-01]])
This is y1[0]: tensor([0.0003, 0.0003, 0.0003, 0.0003])
This is y1[1]: tensor([0.0180, 0.0180, 0.0180, 0.0180])
tensor([[0.0321, 0.0871, 0.2369, 0.6439],
        [0.0321, 0.0871, 0.2369, 0.6439],
        [0.0321, 0.0871, 0.2369, 0.6439]])
This is y2[0]: tensor([0.0321, 0.0871, 0.2369, 0.6439])
This is y2[1]: tensor([0.0321, 0.0871, 0.2369, 0.6439])
```



当然针对数值溢出有其对应的优化方法，将每一个输出值减去输出值中最大的值。

![[公式]](https://www.zhihu.com/equation?tex=softmax%28z_%7Bi%7D%29%3D%5Cfrac%7Be%5E%7Bz_%7Bi%7D+-+D%7D%7D%7B%5Csum_%7Bc+%3D+1%7D%5E%7BC%7D%7Be%5E%7Bz_%7Bc%7D-D%7D%7D%7D)

#### 8.2 about 高纬度下torch.nn.functional.softmax() 函数

```python
import torch

import torch.nn.functional as F

input = np.random.randint(0, 10, size=(2,3,4,5))

input = input.astype(float) # 这里因为softmax不允许int型输入，所以需要转换一下.

input = torch.from_numpy(input) # softmax只接受tensor，所以还需要转换一下

print(input)



a = F.softmax(input,dim=0) #计算第1维度的sofrmax(从左边数)

b = F.softmax(input,dim=1) #计算第2维度的sofrmax

c = F.softmax(input,dim=2) #计算第3维度的softmax

d = F.softmax(input,dim=-1) #计算最后一个维度的softmax 

print("when dim is 0 :",a)

print("when dim is 1 :",b)

print("when dim is 2 :",c)

print("when dim is -1 :",d)
```



#### 9. about item() from pytorch

文档中给了例子，说是一个元素张量可以用item得到元素值，请注意这里的print(x)和print(x.item())值是不一样的，一个是打印张量，一个是打印元素：

```bash
x = torch.randn(1)
print(x)
print(x.item())

#结果是
tensor([-0.4464])
-0.44643348455429077
```



#### 10. about torch.sort() function

***注意这里面的tensor一定要注意维度***

```python
logits = torch.tensor(

  [

​    [

​      [-0.5816, -0.3873, -1.0215, -1.0145, 0.4053],

​      [0.7265, 1.4164, 1.3443, 1.2035, 1.8823],

​      [-0.4451, 0.1673, 1.2590, -2.0757, 1.7255],

​      [0.2021, 0.3041, 0.1383, 0.3849, -1.6311],

​    ]

  ]

)



sorted_logits, sorted_indices = torch.sort(

  logits, descending=True, dim=-1

) # dim=-1 按照行排序

print("according to the line: ", sorted_logits)

print("according to the line: ", sorted_indices)



sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1) # 按照列排序

print("according to the column: ", sorted_logits)

print("according to the column: ", sorted_indices)
```



output:

```powershell
according to the line:  tensor([[[ 0.4053, -0.3873, -0.5816, -1.0145, -1.0215],
         [ 1.8823,  1.4164,  1.3443,  1.2035,  0.7265],
         [ 1.7255,  1.2590,  0.1673, -0.4451, -2.0757],
         [ 0.3849,  0.3041,  0.2021,  0.1383, -1.6311]]])
according to the line:  tensor([[[4, 1, 0, 3, 2],
         [4, 1, 2, 3, 0],
         [4, 2, 1, 0, 3],
         [3, 1, 0, 2, 4]]])
according to the column:  tensor([[[ 0.7265,  1.4164,  1.3443,  1.2035,  1.8823],
         [ 0.2021,  0.3041,  1.2590,  0.3849,  1.7255],
         [-0.4451,  0.1673,  0.1383, -1.0145,  0.4053],
         [-0.5816, -0.3873, -1.0215, -2.0757, -1.6311]]])
according to the column:  tensor([[[1, 1, 1, 1, 1],
         [3, 3, 2, 3, 2],
         [2, 2, 3, 0, 0],
         [0, 0, 0, 2, 3]]])
```



#### 11. about **torch.randn()**  torch.mean() torch.pow() torch.matmul() torch.ones_like()

产生大小为指定的，正态分布的采样点，数据类型是tensor

**torch.mean()**

torch.mean(input) 输出input 各个元素的的均值，不指定任何参数就是所有元素的算术平均值，指定参数可以计算每一行或者 每一列的算术平均数



```python
a=torch.randn(3)  #生成一个一维的矩阵
b=torch.randn(1,3)  #生成一个二维的矩阵
print(a)
print(b)
torch.mean(a)
```

output:

```powershell
tensor([-1.0737, -0.8689, -0.9553])
tensor([[-0.4005, -0.6812,  0.0958]])

tensor(-0.9659)
```

如果指定参数的话，

keepdim=True, input 和output将保持一样的size. otherwise,dim is squeezed. 默认是false.

```python
a = torch.randn(4, 5)  # 4行5列
print(a)
c = torch.mean(a, dim=0, keepdim=True)  # dim为0时 求列的平均值
print("This is mean of column: ", c)
d = torch.mean(a, dim=1, keepdim=True)  # dim为1时 求行的平均值
print("This is mean of line: ", d)
```

output:

```powershell
tensor([[ 0.8913, -0.4275, -0.5879,  0.3359,  1.4847],
        [-0.6798,  0.6409, -0.4618,  0.2167,  0.0430],
        [-0.9019,  0.8588, -1.4371,  0.7955, -0.2983],
        [-0.2362,  1.3501,  0.0100,  0.4004, -0.2030]])
This is mean of column:  tensor([[-0.2317,  0.6056, -0.6192,  0.4371,  0.2566]])
This is mean of line:  tensor([[ 0.3393],
        [-0.0482],
        [-0.1966],
        [ 0.2643]])
```



**torch.pow()**

```python
a = torch.tensor(3)

b = torch.pow(a, 2)

print("This is after pow: ", b.item())

c = torch.randn(4)

print("This is c: ", c)

d = torch.pow(c, 2)

print("This is after pow: ", d)
```

output

```powershell
This is after pow:  9
This is c:  tensor([-0.7659, -0.5344,  0.1736,  1.1546])
This is after pow:  tensor([0.5866, 0.2855, 0.0301, 1.3331])
```



**torch.matmul()**

torch.matmul 是做矩阵乘法

```python
a=torch.tensor([1,2,3])
b=torch.tensor([3,4,5])
torch.matmul(a, b)
```

output:

```powershell
tensor(26)
```



#### 12. about hook

hook种类分为两种:

Tensor级别  register_hook(hook) ->为Tensor注册一个backward hook，用来获取变量的梯度；hook必须遵循如下的格式：hook(grad) -> Tensor or None

nn.Module对象 register_forward_hook(hook)和register_backward_hook(hook)两种方法，分别对应前向传播和反向传播的hook函数。


hook作用：

获取某些变量的中间结果的。**Pytorch会自动舍弃图计算的中间结果**，所以想要获取这些数值就需要使用hook函数。hook函数在使用后应及时删除，以避免每次都运行钩子增加运行负载。



#### 13. pytorch.data属性和.detach()属性相同与不同之处

example: .data 和.detach()只取出本体tensor数据，舍弃了grad，grad_fn等额外反向图计算过程需保存的额外信息。

```python
a = torch.tensor([1.,2,3], requires_grad = True)

print(a)

b = a.data

print(b)

c = a.detach()

print(c)



b *= 5

print("This is a: ", a)

print("This is b: ", b)

print("This is c: ", c)



c *= 4

print("This is a: ", a)

print("This is b: ", b)

print("This is c: ", c)
```



output:

```powershell
tensor([1., 2., 3.], requires_grad=True) 

tensor([1., 2., 3.]) 

tensor([1., 2., 3.]) 

This is a:  tensor([ 5., 10., 15.], requires_grad=True) 

This is b:  tensor([ 5., 10., 15.]) 

This is c:  tensor([ 5., 10., 15.]) 



This is a:  tensor([20., 40., 60.], requires_grad=True) 

This is b:  tensor([20., 40., 60.]) 

This is c:  tensor([20., 40., 60.])
```

**简单的说就是，.data取出本体tensor后仍与原数据共享内存（从第一个代码段中可以看出），在使用in-place操作后，会修改原数据的值，而如果在反向传播过程中使用到原数据会导致计算错误，而使用.detach后，如果在“反向传播过程中”发现原数据被修改过会报错。更加安全**



#### 14. PyTorch中in-place

in-place operation在pytorch中是指改变一个tensor的值的时候，不经过复制操作，而是直接在原来的内存上改变它的值。可以把它成为原地操作符。

在pytorch中经常加后缀“_”来代表原地 in-place operation，比如说.add_() 或者.scatter_()。python里面的+=，*=也是in-place operation。


```python
import torch
x=torch.rand(2) #tensor([0.8284, 0.5539])
print(x)
y=torch.rand(2)
print(x+y)      #tensor([1.0250, 0.7891])
print(x)        #tensor([0.8284, 0.5539])
```



```python
import torch
x=torch.rand(2) #tensor([0.8284, 0.5539])
print(x)
y=torch.rand(2)
x.add_(y)
print(x)        #tensor([1.1610, 1.3789])
```



#### 15. PyTorch笔记之 scatter() 函数

**scatter()** 和 **scatter_()** 的作用是一样的，只不过 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会

> PyTorch 中，一般函数加**下划线**代表直接在原来的 Tensor 上修改

scatter(dim, index, src) 的参数有 3 个

- **dim：**沿着哪个维度进行索引
- **index：**用来 scatter 的元素索引
- **src：**用来 scatter 的源元素，可以是一个标量或一个张量

> 这个 scatter 可以理解成放置元素或者修改元素

简单说就是通过一个张量 src 来修改另一个张量，哪个元素需要修改、用 src 中的哪个元素来修改由 dim 和 index 决定

官方文档给出了 3维张量 的具体操作说明，如下所示

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

example:

```python
x = torch.rand(2, 5)

#tensor([[0.1940, 0.3340, 0.8184, 0.4269, 0.5945],
#        [0.2078, 0.5978, 0.0074, 0.0943, 0.0266]])

torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

#tensor([[0.1940, 0.5978, 0.0074, 0.4269, 0.5945],
#        [0.0000, 0.3340, 0.0000, 0.0943, 0.0000],
#        [0.2078, 0.0000, 0.8184, 0.0000, 0.0266]])
```

如果是二维的例子，则应该对应下面的情况：

```python
y = y.scatter(dim,index,src)
 
#则：
y [ index[i][j] ] [j] = src[i][j] #if dim==0
y[i] [ index[i][j] ]  = src[i][j] #if dim==1 
```

那么这个函数有什么作用呢？其实可以利用这个功能将pytorch 中mini batch中的返回的label（特指[ 1,0,4,9 ]，即size为[4]这样的label）转为one-hot类型的label,举例子如下：

```python
import torch

mini_batch = 4
out_planes = 6
out_put = torch.rand(mini_batch, out_planes)
print("This is old out_put:\n" , out_put)
softmax = torch.nn.Softmax(dim=1)
out_put = softmax(out_put)
 
print("This is new out_put: \n", out_put)
label = torch.tensor([1,3,3,5])
one_hot_label = torch.zeros(mini_batch, out_planes).scatter_(1,label.unsqueeze(1),1)
# index[i][j] 有两个维度，所以这里需要扩容.
print(one_hot_label)
```

output:

```powershell
This is old out_put: 

tensor([[0.7513, 0.4557, 0.2814, 0.4894, 0.6988, 0.9134],        

[0.3404, 0.9000, 0.4107, 0.8312, 0.2615, 0.1938],        

[0.4149, 0.9932, 0.1023, 0.5605, 0.3504, 0.3599],       

[0.1629, 0.3053, 0.4270, 0.2092, 0.5538, 0.3117]]) 

This is new out_put:  

tensor([[0.1900, 0.1414, 0.1187, 0.1462, 0.1803, 0.2234],        

[0.1380, 0.2416, 0.1481, 0.2255, 0.1276, 0.1192],        

[0.1525, 0.2720, 0.1116, 0.1765, 0.1430, 0.1444],        

[0.1400, 0.1615, 0.1824, 0.1467, 0.2070, 0.1625]]) 

tensor([[0., 1., 0., 0., 0., 0.],        

[0., 0., 0., 1., 0., 0.],        

[0., 0., 0., 1., 0., 0.],        

[0., 0., 0., 0., 0., 1.]])
```



#### 16. torch.nn.ReLU

```tex
参数inplace=True:
inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
inplace：can optionally do the operation in-place. Default: False
注： 产生的计算结果不会有影响。利用in-place计算可以节省内"RAM(Random-access memory)"（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。
```

```python
import torch
import torch.nn as nn

input = torch.randn(5)
print('输入处理前:\n', input, input.size())
print('*'*70)

print("Default. inplace=False:")
output_F = nn.ReLU(inplace=False)(input)
print('输入:\n', input, input.size())
print('输出:\n', output_F, output_F.size())

print('*'*70)

print("inplace=True:")
output_T = nn.ReLU(inplace=True)(input)
print('输入:\n', input, input.size())
print('输出:\n', output_T, output_T.size())
```

 output:

```powershell
输入处理前:
 tensor([-1.5561, -1.3829, -0.7814, -0.4832,  0.1552]) torch.Size([5])
**********************************************************************
Default. inplace=False:
输入:
 tensor([-1.5561, -1.3829, -0.7814, -0.4832,  0.1552]) torch.Size([5])
输出:
 tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.1552]) torch.Size([5])
**********************************************************************
inplace=True:
输入:
 tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.1552]) torch.Size([5])
输出:
 tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.1552]) torch.Size([5])
```



#### 17. pytorch 中retain_graph==True的作用

总的来说进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph==True后,可以再来一次backward。

官方定义:

retain_graph (bool, optional) – If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

大意是如果设置为False，计算图中的中间变量在计算完后就会被释放。但是在平时的使用中这个参数默认都为False从而提高效率，和creat_graph的值一样。

具体看一个例子理解：

假设一个我们有一个输入x，y = x **2, z = y*4，然后我们有两个输出，一个output_1 = z.mean()，另一个output_2 = z.sum()。然后我们对两个output执行backward。

```python
import torch
x = torch.randn((1,4),dtype=torch.float32,requires_grad=True)
y = x ** 2
z = y * 4
print("This is s:\n", x)
print("This is y:\n", y)
print("This is z:\n", z)
loss1 = z.mean()
loss2 = z.sum()
print(loss1,loss2)
loss1.backward()    # 这个代码执行正常，但是执行完中间变量都free了，所以下一个出现了问题
print(loss1,loss2)
loss2.backward()    # 这时会引发错误
```

程序正常执行到第12行，所有的变量正常保存。但是在第13行报错：

RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

分析：计算节点数值保存了，但是计算图x-y-z结构被释放了，而计算loss2的backward仍然试图利用x-y-z的结构，因此会报错。

因此需要retain_graph参数为True去保留中间参数从而两个loss的backward()不会相互影响。正确的代码应当把第11行以及之后改成

```python
1 # 假如你需要执行两次backward,先执行第一个的backward，再执行第二个backward
2 loss1.backward(retain_graph=True)# 这里参数表明保留backward后的中间参数。
3 loss2.backward() # 执行完这个后，所有中间变量都会被释放，以便下一次的循环
4  #如果是在训练网络optimizer.step() # 更新参数
```



#### 18. clone() 与 detach() 对比

Torch 为了提高速度，向量或是矩阵的赋值是指向同一内存的，这不同于 Matlab。

如果需要保存旧的tensor即需要开辟新的存储地址而不是引用，可以用 clone() 进行**深拷贝**，
首先我们来打印出来clone()操作后的数据类型定义变化：

**(1). 简单打印类型**

```python
import torch

a = torch.tensor(1.0, requires_grad=True)
b = a.clone()
c = a.detach()
a.data *= 3
b += 1

print(a)   # tensor(3., requires_grad=True)
print(b)
print(c)

'''
输出结果：
tensor(3., requires_grad=True)
tensor(2., grad_fn=<AddBackward0>)
tensor(3.)      # detach()后的值随着a的变化出现变化
'''
```

grad_fn=<CloneBackward>，表示clone后的返回值是个中间变量，因此支持梯度的回溯。clone操作在一定程度上可以视为是一个identity-mapping函数。
detach()操作后的tensor与原始tensor共享数据内存，当原始tensor在计算图中数值发生反向传播等更新之后，detach()的tensor值也发生了改变。
注意： 在pytorch中我们不要直接使用id是否相等来判断tensor是否共享内存，这只是充分条件，因为也许底层共享数据内存，但是仍然是新的tensor，比如detach()，如果我们直接打印id会出现以下情况。

```python
import torch as t
a = t.tensor([1.0,2.0], requires_grad=True)
b = a.detach()
#c[:] = a.detach()
print(id(a))
print(id(b))
#140568935450520
140570337203616
```

显然直接打印出来的id不等，我们可以通过简单的赋值后观察数据变化进行判断。

**(2). clone()的梯度回传**



#### 19. pytorch view()

1.在PyTorch中**view**函数作用为重构张量的维度，相当于numpy中的resize()的功能，但是用法不太一样.

tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。

```python
import torch
tt1=torch.tensor([-0.3623,-0.6115,0.7283,0.4699,2.3261,0.1599])
result=tt1.view(3,2)
result
```

则`tt1.size()`为`torch.Size([6])`，是一个一行的tensor。现在通过view可以将其重构一下形状。

output:

```powershell
tensor([[-0.3623, -0.6115],
        [ 0.7283,  0.4699],
        [ 2.3261,  0.1599]])
```

2.有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况。

```python
import torch

tt2=torch.tensor([[-0.3623, -0.6115],[ 0.7283, 0.4699],[ 2.3261, 0.1599]])

result=tt2.view(-1)

result
```

```powershell
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599]) 
```

由上面的案例可以看到，如果是torch.view(-1)，则原张量会变成一维的结构。

```python
import torch

tt3=torch.tensor([[-0.3623, -0.6115],[ 0.7283, 0.4699],[ 2.3261, 0.1599]])

result=tt3.view(2,-1)

result

```

```powershell
tensor([[-0.3623, -0.6115,  0.7283],        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。

则为两行.



当-1在左边时，会变成两列.

```python
import torch

b=a.view(-1,2)  #当-1在左边时，会变成两列.

print(b)
```



#### 20. pytorch **Matmul 函数**

张量相乘函数

```python
import torch

# a为1D张量,b为2D张量
a = torch.tensor([1., 2.])
b = torch.tensor([[5., 6., 7.], [8., 9., 10.]])

result = torch.matmul(a, b)
print(result.size())
# torch.Size([3])

print(result)
# tensor([21., 24., 27.])
```

![qwj0ottute](C:\Users\34890\Pictures\Camera Roll\qwj0ottute.png)

#### 21. pytorch nn.MSELoss() 均方损失函数

```python
import torch
import torch.nn as nn
crit=nn.MSELoss()#均方损失函数
target = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
pred= torch.FloatTensor([[7, 8, 9], [8, 4, 3]])
cost=crit(pred,target)#将pred,target逐个元素求差,然后求平方,再求和,再求均值,
print(cost)#tensor(22.3333)
sum=0
for i in range (0,2):#遍历行i
    for j in range(0,3):#遍历列
        sum+=(target[i][j]-pred[i][j])*(target[i][j]-pred[i][j])#对应元素做差,然后平方
print(sum/6)#tensor(22.3333)
```

很多的loss函数都有size_average和reduce两个布尔类型的参数，因为一般损失函数都是直接计算batch的数据，因此返回的loss结果都是维度为(batch_size,)的向量。

1）如果reduce=False,那么size_average参数失效，直接返回向量形式的loss

2)如果redcue=true,那么loss返回的是标量。

   2.a: if size_average=True, 返回loss.mean();#就是平均数

   2.b: if size_average=False,返回loss.sum()

注意：默认情况下，reduce=true,size_average=true



#### 22. Python getattr() 函数 

描述

**getattr()** 函数用于返回一个对象属性值。

语法

```python
getattr(object, name[, default])
```

##### 参数

- object -- 对象。
- name -- 字符串，对象属性。
- default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。

##### Example 1: How getattr() works in Python?

```python
class Person:
    age = 23
    name = "Adam"

person = Person()
print('The age is:', getattr(person, "age"))
print('The age is:', person.age)
```

**Output**

```powershell
The age is: 23
The age is: 23
```

------

##### Example 2: getattr() when named attribute is not found

```python
class Person:
    age = 23
    name = "Adam"

person = Person()

# when default value is provided
print('The sex is:', getattr(person, 'sex', 'Male'))

# when no default value is provided
print('The sex is:', getattr(person, 'sex'))
```

**Output**

```powershell
The sex is: Male
AttributeError: 'Person' object has no attribute 'sex'
```



#### 23. python hasattr()

The hasattr() method returns true if an object has the given named attribute and false if it does not.

The syntax of `hasattr()` method is:

```
hasattr(object, name)
```

`hasattr()` is called by [getattr()](https://www.programiz.com/python-programming/methods/built-in/getattr) to check to see if AttributeError is to be raised or not.

------

##### hasattr() Parameters

`hasattr()` method takes two parameters:

- **object** - object whose named attribute is to be checked
- **name** - name of the attribute to be searched

------

##### Return value from hasattr()

`hasattr()` method returns:

- **True**, if object has the given named attribute
- **False**, if object has no given named attribute

------

##### Example: How hasattr() works in Python?

```python
class Person:
    age = 23
    name = 'Adam'

person = Person()

print('Person has age?:', hasattr(person, 'age'))
print('Person has salary?:', hasattr(person, 'salary'))
```

**Output**

```powershell
Person has age?: True
Person has salary?: False
```



#### 24. Python Anonymous [əˈnɒnɪməs]  [əˈnɑːnɪməs] /Lambda Function

##### 1 for 简写

先举一个例子：

```python
y = [1,2,3,4,5,6]
[(i*2) for i in y ]
```

会输出  [2, 4, 6, 8, 10, 12]

##### 1.1 一层for循环简写：

一层 for 循环的简写格式是：（注意有中括号）

```python
[ 对i的操作 for i in 列表 ]

```

它相当于：

```python
for i in 列表:
    对i的操作
```

##### 1.2 两层for循环

两层的for循环就是：

```python
[对i的操作 for 单个元素 in 列表 for i in 单个元素]

```

举个简单的例子：

```python
y_list = ['assss','dvv']
[print(i) for y in y_list for i in y]
```

得到结果：a s s s s d v v

他类似于：

```python
y_list = ['assss','dvv']
for y in y_list:
    for i in y:
        print(i) 
```

##### 2 if 简写

格式是：

```python
True的逻辑 if 条件 else False的逻辑

```

举个例子：

```python
y = 0
x = y+3 if y > 3 else y-1
```

此时 x = -1

因为 y = 0 ，所以判断 y>3 时执行了 False的逻辑：y-1，所以x的值为 -1

##### 2.1 for 与 if 的结合怎么简写

举个l例子：

```python
x = [1,2,3,4,5,6,7]
[print(i) for i in x if i > 3 ]
```

它会输出：4 5 6 7

注：使用简写的方式无法对 if 判断为 False 的对象执行操作。

所以它的模板是：

```python
[判断为True的i的操作 for i in 列表 if i的判断 ]
```

##### 3 匿名函数lambda

匿名函数的使用方法是：

lambda 参数: 表达式
举个例子：

```python
x = 3
(lambda k: k+3)(x)
```

输出 6

这是一个比较简单的匿名函数表达式，一般匿名函数会结合很多其他函数，作为传递参数的作用。

举一个有点儿hard的例子:

```python
      self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
```



```python
print("*******if...else语句*********")
#if 条件为真的时候返回if前面内容，否则返回0 
exp1= lambda x:x+1 if  2==1 else 0 
print(exp1(2))
exp2 = lambda x:x+1 if  1==1 else 0 
print(exp2(2))
```

output:

```powershell
****\**\*if…else语句\*\**\******
0
3
[Finished in 0.2s]
```



```python
print("*******if not...else语句*********")  
#if not 为假返回if not前面内容，否则返回0  
exp3 = lambda x:x+1 if not 2==1 else 0  
print(exp3(2))  

exp4 = lambda x:x+1 if not 1==1 else 0  
print(exp4(2))  
```

```powershell
结果
3
0
[Finished in 0.3s]
```



#### 25. method 与 function的区别

我们在阅读英文资料时，可能经常会遇到method和function这两个单词，还可能经常以为两个是一样的。

这次在读python的说明文档时，这两个词出现的频率挺高，所以我就查了以下它们的区别。

method是依赖与一个对象的，function是独立于对象的。

在c中，只有function;

在c++中，既有method也有function,一个函数的称呼取决于它是否是一个类的对象，同理，python也是，php也是。

在java中，只有method，因为它是一门纯面向对象的语言。

下面是一段 python的代码：

```python
def function(data):
        return data;
class A:
        str1 = "I'm a method in class"
        def method(self):
                return self.str1

str2 = "I'm a function"
print(function(str2))
a = A()
print(a.method()) 
```


从上面中可以看出来两者之间的差别:

1 function是直接通过名字来调用的，它只能被传递参数来处理或者使用全局变量。

2 method 是通过与一个对象相关联的名字来调用的，它既可以被传递参数也可以，使用对象内部的数据。

  method 隐式的被传递了调用它的对象。



#### 26. python underscore

**Python****里的单下划线，双下划线，以及前后都带下划线的意义：**

1. 单下划线如：_name

   意思是：不能通过from modules import * 导入，如需导入需要：from modules import _name

2. 对象前面加双下划线如：__name

   意思是：生命对象为私有

3. 前后双下划线如：__init __:python系统自带的一些函数和方法



#### 27. python中的类型提示(type hint)

在刷leetcode或者一些官方源码的时候，经常看到如下字样：

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
```

这就是类型提示(type hint)，下面来个简单的例子，

```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

如上，其中name是传入的参数，而:右边的str则是name期望的类型即str，而->则指向期望函数的返回类型。
如果不期望有返回值可以直接指向None，如下：

```python
def feeder(get_next_item: Callable[[], str]) -> None:

```



#### 28.关于 from . import 问题

一个点表示当前路径，二个点表示上一层路径。

```python
from . import echo

from .. import formats
```



#### 29.关于 from XXX import * ,import,from XXX import问题

1. from XXX import* 会导入XXX模块的所有函数, 不建议用这个. 
2. import  XXX (here the XXX can't be folder)
2. **import 模块**：导入一个模块；注：相当于导入的是一个文件夹，是个相对路径。
4. **from…import**：导入了一个模块中的一个函数；注：相当于导入的是一个文件夹中的文件，是个绝对路径。



#### 30.关于sys模块

Python sys模块通过sys.argv提供对任何命令行参数的访问。这有两个常用指令：

sys.argv 返回的是包含命令行参数的一个 list

len(sys.argv) 返回的是命令行参数的个数

```python
import sys

print('Number of arguments:'+ len(sys.argv) + ' arguments.')
print('Argument List:', str(sys.argv)
```

cmd 直接以下命令：

```powershell
 python test.py arg1 arg2 arg3

```

可返回如下结果：

```powershell
Number of arguments: 4 arguments.
Argument List: ['test.py', 'arg1', 'arg2', 'arg3']
```

返回的第一个参数永远是文件名，且会被计入参数个数中。



#### 31. for i in range()

```python
for i in range(10,15):

​    print(i)
```

```powershell
10 11 12 13 14
```



#### 32. split() function

##### 描述

split() 通过指定分隔符对字符串进行**切片**，如果第二个参数 num 有指定值，则分割为 num+1 个子字符串。

##### 语法

split() 方法语法：

```python
str.split(str="", num=string.count(str))
```

##### 参数

- str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
- num -- 分割次数。默认为 -1, 即分隔所有。

##### 返回值

返回分割后的字符串列表。

##### 实例

以下实例展示了 split() 函数的使用方法：

##### Example

```python
#!/usr/bin/python3  



str = "this is string example....wow!!!" 

print (str.split( ))       # 以空格为分隔符 

print (str.split('i',1))   # 以 i 为分隔符 

print (str.split('w'))     # 以 w 为分隔符
```

以上实例输出结果如下：

```python
['this', 'is', 'string', 'example....wow!!!']
['th', 's is string example....wow!!!']
['this is string example....', 'o', '!!!']
```

以下实例以 # 号为分隔符，指定第二个参数为 1，返回两个参数列表。

##### Example

```python
#!/usr/bin/python3  

txt = "Google#Runoob#Taobao#Facebook"   

x = txt.split("#", 1)  

print(x)
```

以上实例输出结果如下：

```powershell
['Google', 'Runoob#Taobao#Facebook']
```



#### 33. add_argument函数的metavar参数

add_argument函数的metavar参数，用来控制部分命令行参数的显示，注意：它只是影响部分参数的显示信息，不影响代码内部获取命令行参数的对象。

```python
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', metavar='YYY')
>>> parser.add_argument('bar', metavar='XXX')
>>> parser.parse_args('X --foo Y'.split())
Namespace(bar='X', foo='Y')
>>> parser.print_help()
usage:  [-h] [--foo YYY] XXX

positional arguments:
 XXX

optional arguments:
 -h, --help  show this help message and exit
 --foo YYY
```



metavar参数可以让命令的帮助信息更好看一些！

初次之外，还有个功能可以关注，对于有nargs参数的命令行参数，可以用metavar来设置每一个具体的参数的名称：

```python
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-x', nargs=2)
>>> parser.add_argument('--foo', nargs=2, metavar=('bar', 'baz'))
>>> parser.print_help()
usage: PROG [-h] [-x X X] [--foo bar baz]

optional arguments:
 -h, --help     show this help message and exit
 -x X X
 --foo bar baz
```

-x参数没有使用metavar，显示出来的帮助信息就是两个X，而--foo参数也可以接收两个参数，这两个参数的名称就用metavar进行了具体的定义，看起来好多了。本文代码示例都是python官方文档中的。



Q1：*请问博主，第一个位置参数假如说是--max_episode_len,然后也有人写是--max-episode-len,但是他在调用的时候仍然用的是args.max_episode_len，也没报错，请问这个下划线_和-的区别在哪里呢？*

A1：没啥区别，在这里表示同一个意思，-对应_，代码里写的不一样或者都改成一样的都可以



#### 34. add_argument函数的prefix_chars

许多命令行会使用 `-` 当作前缀，比如 `-f/--foo`。如果解析器需要支持不同的或者额外的字符，比如像 `+f` 或者 `/foo` 的选项，可以在参数解析构建器中使用 `prefix_chars=` 参数。

```python
>>> parser = argparse.ArgumentParser(prog='PROG', prefix_chars='-+')
>>> parser.add_argument('+f')
>>> parser.add_argument('++bar')
>>> parser.parse_args('+f X ++bar Y'.split())
Namespace(bar='Y', f='X')
```

`prefix_chars=` 参数默认使用 `'-'`。 提供一组不包括 `-` 的字符将导致 `-f/--foo` 选项不被允许。



#### 35. python var函数

```python
class My():

​    'Test'

​    def __init__(self,name):

​        self.name=name



​    def test(self):

​        print (self.name)



​    def abc(self):

​        print("roll out")




vars(My)#返回一个字典对象，他的功能其实和  My.__dict__  很像

for key,value in vars(My).items():

​    print (key,':',value)
```

output:

```powershell
__module__ : __main__ 
__doc__ : Test 
__init__ : <function My.__init__ at 0x7f21202c0170> 
test : <function My.test at 0x7f21202c0ef0> 
abc : <function My.abc at 0x7f21202c05f0> 
__dict__ : <attribute '__dict__' of 'My' objects> 
__weakref__ : <attribute '__weakref__' of 'My' objects>
```



#### 36.torchvision.transforms.compose()

torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision.transforms主要是用于常见的一些图形变换。以下是torchvision的构成：

torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。

##### 1.transforms.Compose()

本文的主题是其中的`torchvision.transforms.Compose()`类。这个类的主要作用是串联多个图片变换的操作。这个类的构造很简单:

- 即组合几个变换方法，按顺序变换相应数据。
- 其中torchscript为脚本模块，用于封装脚本跨平台使用，若需要支持此情况，需要使用torch.nn.Sequential，而不是compose
- 对应于问题描述中代码，即先应用ToTensor()使[0-255]变换为[0-1]，再应用Normalize自定义标准化



```python
class torchvision.transforms.Compose(transforms):

 # Composes several transforms together.

 # Parameters: transforms (list of Transform objects) – list of transforms to compose.

Example # 可以看出Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。

>>> transforms.Compose([
>>>  transforms.CenterCrop(10),
>>>  transforms.ToTensor(),])


```

事实上，`Compose()`类会将transforms列表里面的transform操作进行遍历。实现的代码很简单：

```python
## 这里对源码进行了部分截取。
def __call__(self, img):
	for t in self.transforms:	
		img = t(img)
    return img

```



##### 2.transforms.ToTensor()

Convert a `PIL Image` or `numpy.ndarray` to tensor
转换一个PIL库的图片或者numpy的数组为tensor张量类型；

转换从[0,255]->[0,1]

- 实现原理，即针对不同类型进行处理，原理即各值除以255，最后通过`torch.from_numpy`将`PIL Image` or `numpy.ndarray`针对具体数值类型比如Int32,int16,float等转成`torch.tensor`数据类型
- **需要注意的是，源码中有一小段内容：**

```python
 if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


```

我们可以看到在转换过程中有一个轴的转置操作`pic.transpose((2, 0, 1))` 和`contiguous()` 函数

- `pic.transpose((2, 0, 1))`将第三维轴换到第一个位置，这样做的原因主要是因为PIEimage与torch和numpy数据类型多维参数位置的区别，以下表说明

| 参数              | 含义   |
| ----------------- | ------ |
| torch：(x,y,z)    | x个y*z |
| PIEimage：(x,y,z) | z个x*y |

即三维表示的结构顺序有区别，导致numpy与torch多维转换时需要转置.

| Normalize a tensor image with mean and standard deviation 通过平均值和标准差来标准化一个tensor图像 |
| ------------------------------------------------------------ |
| 公式为： output[channel] = (input[channel] - mean[channel]) / std[channel] |

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))解释：

第一个(0.5,0.5,0.5) 即三个通道的平均值
第二个(0.5,0.5,0.5) 即三个通道的标准差值
由于ToTensor()已经将图像变为[0,1]，我们使其变为[-1,1]，以第一个通道为例，将最大与最小值代入公式

(0-0.5)/0.5=-1
(1-0.5)/0.5=1
其他数值同理操作，即映射到[-1,1]

#### 37. python __getitem__ ()

__getitem__() is a magic method in Python, which when used in a class, **allows its instances to use the [] (indexer) operators**. 

```python
class P(object):

​    def __init__(self):

​        self.cus_dict = {

​            'name': 'abc'

​        }

​        print("函数经过初始化")



​    def __getitem__(self, item):

​        print("函数经过getitem")

​        return self.cus_dict[item]



if __name__ == '__main__':

​    p = P()

​    print(p['name'])
```



#### 38. def ____len____(self):

```python
class Fib(object):

​    def __init__(self, num):

​        a, b, L = 0, 1, []

​        for n in range(num):

​            L.append(a)

​            a, b = b, a + b

​        self.numbers = L



​    def __str__(self):

​        return str(self.numbers)



​    __repr__ = __str__



​    def __len__(self):

​        print("code goes here")

​        return len(self.numbers)


f = Fib(10)

print (f)

print (len(f))
```

output:

```powershell
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 

code goes here 

10
```



#### 39. torch.utils.data.DataLoader()

DataLoader是PyTorch中的一种数据类型。

在PyTorch中训练模型经常要使用它，那么该数据结构长什么样子，如何生成这样的数据类型？

下面就研究一下：

先看看 dataloader.py脚本是怎么写的（VS中按F12跳转到该脚本）

 __init__（构造函数）中的几个重要的属性：

1、dataset：（数据类型 dataset）

输入的数据类型。看名字感觉就像是数据库，C#里面也有dataset类，理论上应该还有下一级的datatable。这应当是原始数据的输入。PyTorch内也有这种数据结构。这里先不管，估计和C#的类似，这里只需要知道是输入数据类型是dataset就可以了。

2、batch_size：（数据类型 int）

每次输入数据的行数，默认为1。PyTorch训练模型时调用数据不是一行一行进行的（这样太没效率），而是一捆一捆来的。这里就是定义每次喂给神经网络多少行数据，如果设置成1，那就是一行一行进行（个人偏好，PyTorch默认设置是1）。

3、shuffle：（数据类型 bool）

洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。

4、collate_fn：（数据类型 callable，没见过的类型）

将一小段数据合并成数据列表，默认设置是False。如果设置成True，系统会在返回前会将张量数据（Tensors）复制到CUDA内存中。（不太明白作用是什么，就暂时默认False）

5、batch_sampler：（数据类型 Sampler）

批量采样，默认设置为None。但每次返回的是一批数据的索引（注意：不是数据）。其和batch_size、shuffle 、sampler and drop_last参数是不兼容的。我想，应该是每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质。所以，它和一捆一捆按顺序输入，数据洗牌，数据采样，等模式是不兼容的。

6、sampler：（数据类型 Sampler）

采样，默认设置为None。根据定义的策略从数据集中采样输入。如果定义采样规则，则洗牌（shuffle）设置必须为False。

7、num_workers：（数据类型 Int）

工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。注意：这个数字必须是大于等于0的，负数估计会出错。

8、pin_memory：（数据类型 bool）

内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。

9、drop_last：（数据类型 bool）

丢弃最后数据，默认为False。设置了 batch_size 的数目后，最后一批数据未必是设置的数目，有可能会小些。这时你是否需要丢弃这批数据。

10、timeout：（数据类型 numeric）

超时，默认为0。是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。 所以，数值必须大于等于0。

11、worker_init_fn（数据类型 callable，没见过的类型）

子进程导入模式，默认为Noun。在数据导入前和步长结束后，根据工作子进程的ID逐个按顺序导入数据。

从DataLoader类的属性定义中可以看出，这个类的作用就是实现数据以什么方式输入到什么网络中。
代码一般是这么写的：

定义学习集 DataLoader

train_data = torch.utils.data.DataLoader(各种设置...) 

将数据喂入神经网络进行训练

for i, (input, target) in enumerate(train_data): 
    循环代码行......

 

如果全部采用默认设置输入数据，数据就是一行一行按顺序输入到神经网络。如果对数据的输入有特殊要求。

比如：想打乱一下数据的排序，可以设置 shuffle（洗牌）为True；

比如：想数据是一捆的输入，可以设置 batch_size 的数目；

比如：想随机抽取的模式输入，可以设置 sampler 或 batch_sampler。如何定义抽样规则，可以看sampler.py脚本。这里不是重点；

比如：像多线程输入，可以设置 num_workers 的数目；

其他的就不太懂了，以后实际应用时碰到特殊要求再研究吧。



```python
from torch.utils.data import DataLoader, Dataset
import torch

class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):

        # 两边都是输出相同索引的数值,根据输出栏目可以看到data和taeget是一套输出的
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)
print("This is data_tensor: ",data_tensor)
print("This is target_tensor: ",target_tensor)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print ('tensor_data[0]: ', tensor_dataset[0])

# 可返回数据len
print ('len os tensor_dataset: ', len(tensor_dataset))

# 这个函数控制着接下来for循环的输出“__getitem__”方法
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出,这里会直接调用
for data, target in tensor_dataloader: 
    print("This is data: ",data)
    print("This is target: ",target)

# 输出一个batch
print ('one batch tensor data: ', iter(tensor_dataloader).next())

# 输出batch数量
print ('len of batchtensor: ', len(list(iter(tensor_dataloader))))

 
```



output:

```python
This is data_tensor:  tensor([[ 1.0177,  0.1941,  0.9400],        [-1.0706, -0.1892,  0.1676],        [ 2.0725, -2.3246,  1.2797],        [-1.2711, -0.0427,  0.4700]]) 
This is target_tensor:  tensor([0.5486, 0.5824, 0.4074, 0.3199]) 
tensor_data[0]:  (tensor([1.0177, 0.1941, 0.9400]), tensor(0.5486)) 
len os tensor_dataset:  4 
This is data:  tensor([[-1.0706, -0.1892,  0.1676],        [ 2.0725, -2.3246,  1.2797]]) 
This is target:  tensor([0.5824, 0.4074]) 
This is data:  tensor([[ 1.0177,  0.1941,  0.9400],        [-1.2711, -0.0427,  0.4700]]) 
This is target:  tensor([0.5486, 0.3199]) 
one batch tensor data:  [tensor([[-1.2711, -0.0427,  0.4700],        [-1.0706, -0.1892,  0.1676]]), tensor([0.3199, 0.5824])] 
len of batchtensor:  2
```



#### 40. tensor().size()

>```python
>(base) PS C:\Users\chenxuqi> python
>Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
>Type "help", "copyright", "credits" or "license" for more information.
>
>>>> import torch
>>>> a = torch.ones(2,3,4)
>>>> a
>>>> tensor([[[1., 1., 1., 1.],
>>>>  [1., 1., 1., 1.],
>>>>  [1., 1., 1., 1.]],
>
>        [[1., 1., 1., 1.],
>         [1., 1., 1., 1.],
>         [1., 1., 1., 1.]]])
>
>>>> a.size
>>>> <built-in method size of Tensor object at 0x000001D0D5CAE368>
>>
>>>> a.size()
>>>> torch.Size([2, 3, 4])
>>
>>>> a.size(0)
>>>> 2
>>>> a.size()[0]
>>>> 2
>>
>>
>>>> a = torch.ones(2,3,4,5,6,7,8,9)
>>>> a.shape
>>>> torch.Size([2, 3, 4, 5, 6, 7, 8, 9])
>>>> a.size()
>>>> torch.Size([2, 3, 4, 5, 6, 7, 8, 9])
>>>> a.size(0)
>>>> 2
>>>> a.size(7)
>>>> 9
>>>> a.size()[0]
>>>> 2
>>>> a.size()[7]
>>>> 9
>>>> a.size(4)
>>>> 6
>>>> a.size()[4]
>>>> 6
>>
>>
>>>> a.size()[:]
>>>> torch.Size([2, 3, 4, 5, 6, 7, 8, 9])
>>
>>>> a.size()[4:]
>>>> torch.Size([6, 7, 8, 9])
>>
>>
>>
>>
>```
>
>

#### 41. tensor转成numpy的几种情况



1. GPU中的Variable变量：

a.cuda().data.cpu().numpy()

\2. GPU中的tensor变量：

a.cuda().cpu().numpy()

\3. CPU中的Variable变量：
a.data.numpy()

\4. CPU中的tensor变量：

a.numpy()

总结：

.cuda()是读取GPU中的数据

.data是读取Variable中的tensor

.cpu是把数据转移到cpu上

.numpy()把tensor变成numpy



#### 42. about grad

```python
import torch



a = torch.tensor([1, 2, 3.], requires_grad=True)

print(a.grad)

out = a.sigmoid()

print("This is out: ",out)

print(out.sum())

out.sum().backward()

\# grad can be implicitly created only for scalar outputs

print(a.grad)
```

output:

```powershell
None 
This is out:  tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>) 
tensor(2.5644, grad_fn=<SumBackward0>) 
tensor([0.1966, 0.1050, 0.0452])
```



#### 43. about register_forward_hook(hook)

register_forward_hook(hook) 最大的作用也就是当训练好某个model，想要展示某一层对最终目标的影响效果。



#### 44. torch.transpose()

这个函数作用是交换两个维度的内容，类似矩阵的转置.

```python
import torch

cc=torch.randn((2,2,3,4))

dd=torch.transpose(cc,1,2)

ee=torch.transpose(cc,0,1)

print("This is the cc:",cc,cc.shape)

print("This is the dd:",dd,dd.shape)

print("This is the ee:",ee,ee.shape)
```



#### 45. torch.arange()

```python
import torch

a=torch.arange(0,24).view(2, 3, 4)

print("This is input: ",a)
```

output:

```powershell
This is input:  tensor([[[ 0,  1,  2,  3],         [ 4,  5,  6,  7],         [ 8,  9, 10, 11]],         [[12, 13, 14, 15],         [16, 17, 18, 19],         [20, 21, 22, 23]]])
```



#### 46. torch change to numpy

```python
import torch

a=torch.arange(0,24).view(2, 3, 2,2)

a=a.numpy()
```



#### 47. torch.flatten()与torch.nn.Flatten()

 [torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).flatten(x)等于torch.flatten(x，0)默认将张量拉成一维的向量，也就是说从第一维开始平坦化，torch.flatten(x，1)代表从第二维开始平坦化。

```python
import torch
x=torch.randn(2,4,2)
print(x)

z=torch.flatten(x)
print(z)

w=torch.flatten(x,1)
print(w)

输出为：
tensor([[[-0.9814,  0.8251],
         [ 0.8197, -1.0426],
         [-0.8185, -1.3367],
         [-0.6293,  0.6714]],

        [[-0.5973, -0.0944],
         [ 0.3720,  0.0672],
         [ 0.2681,  1.8025],
         [-0.0606,  0.4855]]])

tensor([-0.9814,  0.8251,  0.8197, -1.0426, -0.8185, -1.3367, -0.6293,  0.6714,
        -0.5973, -0.0944,  0.3720,  0.0672,  0.2681,  1.8025, -0.0606,  0.4855])

tensor([[-0.9814,  0.8251,  0.8197, -1.0426, -0.8185, -1.3367, -0.6293,  0.6714]
,
        [-0.5973, -0.0944,  0.3720,  0.0672,  0.2681,  1.8025, -0.0606,  0.4855]
])


```

 torch.flatten(x,0,1)代表在第一维和第二维之间平坦化

```python
import torch
x=torch.randn(2,4,2)
print(x)
 
w=torch.flatten(x,0,1) #第一维长度2，第二维长度为4，平坦化后长度为2*4
print(w.shape)
 
print(w)
 
输出为：
tensor([[[-0.5523, -0.1132],
         [-2.2659, -0.0316],
         [ 0.1372, -0.8486],
         [-0.3593, -0.2622]],
 
        [[-0.9130,  1.0038],
         [-0.3996,  0.4934],
         [ 1.7269,  0.8215],
         [ 0.1207, -0.9590]]])
 
torch.Size([8, 2])
 
tensor([[-0.5523, -0.1132],
        [-2.2659, -0.0316],
        [ 0.1372, -0.8486],
        [-0.3593, -0.2622],
        [-0.9130,  1.0038],
        [-0.3996,  0.4934],
        [ 1.7269,  0.8215],
        [ 0.1207, -0.9590]])
 
```

对于torch.nn.Flatten()，因为其被用在神经网络中，输入为一批数据，第一维为batch，通常要把一个数据拉成一维，而不是将一批数据拉为一维。所以torch.nn.Flatten()默认从第二维开始平坦化。

```python
import torch
#随机32个通道为1的5*5的图
x=torch.randn(32,1,5,5)

model=torch.nn.Sequential(
    #输入通道为1，输出通道为6，3*3的卷积核，步长为1，padding=1
    torch.nn.Conv2d(1,6,3,1,1),
    torch.nn.Flatten()
)
output=model(x)
print(output.shape)  # 6*（7-3+1）*（7-3+1）

输出为：

torch.Size([32, 150])
```



#### 48. optimizer.zero_grad()

![](/home/jiang/桌面/About Python and some image algorithm/pictures source/zero_grad.webp)

#### 49. about with torch.no_grad()

torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。

例如：

```python
a = torch.tensor([1.1], requires_grad=True)
b = a * 2
```

打印b可看到其 grad_fn 为 mulbackward 表示是做的乘法。

```python
b
Out[63]: tensor([2.2000], grad_fn=<MulBackward0>)
```

```python
b.add_(2)
Out[64]: tensor([4.2000], grad_fn=<AddBackward0>)
```

可以看到不被wrap的情况下，b.grad_fn 为 addbackward 表示这个add 操作被track了

```python
with torch.no_grad():
    b.mul_(2)
```


在被包裹的情况下可以看到 b.grad_fn 还是为 add，mul 操作没有被 track. 但是注意，乘法操作是被执行了的。(4.2 -> 8.4)

```python
b
Out[66]: tensor([8.4000], grad_fn=<AddBackward0>)
```


所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。

同时 torch.no_grad() 还可以作为一个装饰器。
比如在网络测试的函数前加上

```python
@torch.no_grad()
def eval():
	...
```


扩展：
同样还可以用 torch.set_grad_enabled()来实现不计算梯度。
例如：

```python
def eval():
	torch.set_grad_enabled(False)
	...	# your test code
	torch.set_grad_enabled(True)
```



#### 50. about torch.cat()

1.字面理解：torch.cat是将两个张量（tensor）拼接在一起，cat是concatenate的意思，即拼接，联系在一起。

2. 例子理解(1)

```python
import torch

A=torch.ones(2,3) #2x3的张量（矩阵）                                     

print(f"This is A:{A}")



B=2*torch.ones(4,3)#4x3的张量（矩阵）                                    

print(f"This is B:{B}")



C=torch.cat((A,B),0)#按维数0（行）拼接

print(f"This is C:{C}")

print(f"This is C size:{C.size()}")



D=2*torch.ones(2,4) #2x4的张量（矩阵）

C=torch.cat((A,D),1)#按维数1（列）拼接，此时的tensor 行数必须一致

print(f"This is C:{C}")

print(f"This is C size:{C.size()}")
```

output:

```bash
This is A:tensor([[1., 1., 1.],        [1., 1., 1.]]) 

This is B:tensor([[2., 2., 2.],        [2., 2., 2.],        [2., 2., 2.],        [2., 2., 2.]]) 

This is C:tensor([[1., 1., 1.],        [1., 1., 1.],        [2., 2., 2.],        [2., 2., 2.],        [2., 2., 2.],        [2., 2., 2.]]) 
This is C size:torch.Size([6, 3]) 

This is C:tensor([[1., 1., 1., 2., 2., 2., 2.], [1., 1., 1., 2., 2., 2., 2.]]) 

This is C size:torch.Size([2, 7])
```



上面给出了两个张量A和B，分别是2行3列，4行3列。即他们都是2维张量。因为只有两维，这样在用torch.cat拼接的时候就有两种拼接方式：按行拼接和按列拼接。即所谓的维数0和维数1. 

C=torch.cat((A,B),0)就表示按维数0（行）拼接A和B，也就是竖着拼接，A上B下。此时需要注意：列数必须一致，即维数1数值要相同，这里都是3列，方能列对齐。拼接后的C的第0维是两个维数0数值和，即2+4=6.

C=torch.cat((A,B),1)就表示按维数1（列）拼接A和B，也就是横着拼接，A左B右。此时需要注意：行数必须一致，即维数0数值要相同，这里都是2行，方能行对齐。拼接后的C的第1维是两个维数1数值和，即3+4=7.

从2维例子可以看出，使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。

**例子理解，区分append函数(2)**

```python
import torch

A=torch.zeros(2,5) #2x5的张量（矩阵）                                     

print(A)

B=torch.ones(3,5)

print(B)

list=[]

list.append(A)

list.append(B)

print(f"This is list{list}")

C=torch.cat(list,dim=0)#按照行进行拼接,此时所有tensor的列数需要相同

print(C,C.shape)
```

output:

```python
tensor([[0., 0., 0., 0., 0.],        [0., 0., 0., 0., 0.]]) 

tensor([[1., 1., 1., 1., 1.],        [1., 1., 1., 1., 1.],        [1., 1., 1., 1., 1.]]) 

This is list[tensor([[0., 0., 0., 0., 0.],        
                     [0., 0., 0., 0., 0.]]), tensor([[1., 1., 1., 1., 1.],        
[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.]])] 

tensor([[0., 0., 0., 0., 0.],        

[0., 0., 0., 0., 0.],        

[1., 1., 1., 1., 1.],        

[1., 1., 1., 1., 1.],        

[1., 1., 1., 1., 1.]]) torch.Size([5, 5])
```



3.实例

在深度学习处理图像时，常用的有3通道的RGB彩色图像及单通道的灰度图。张量size为cxhxw,即通道数x图像高度x图像宽度。在用torch.cat拼接两张图像时一般要求图像大小一致而通道数可不一致，即h和w同，c可不同。当然实际有3种拼接方式，另两种好像不常见。比如经典网络结构：U-Net

​     ![U-Net](/home/jiang/桌面/About Python and some image algorithm/pictures source/U-Net.png)                               

里面用到4次torch.cat,其中copy and crop操作就是通过torch.cat来实现的。可以看到通过上采样（up-conv 2x2）将原始图像h和w变为原来2倍，再和左边直接copy过来的同样h,w的图像拼接。这样做，可以有效利用原始结构信息。

4.总结

使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能






## About python syntax

#### 1.关于python中self的问题

Python编写类的时候，每个函数参数第一个参数都是self，一开始我不管它到底是干嘛的，只知道必须要写上。后来对Python渐渐熟悉了一点，再回头看self的概念，似乎有点弄明白了。

首先明确的是self只有在类的方法中才会有，独立的函数或方法是不必带有self的。self在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。

self名称不是必须的，在python中self不是关键词，你可以定义成a或b或其它名字都可以,但是约定成俗（为了和其他编程语言统一，减少理解难度），不要搞另类，大家会不明白的。

下例中将self改为myname一样没有错误：

```python
lass Person:
    def __init__(myname,name):
        myname.name=name
    def sayhello(myname):
        print ('My name is:',myname.name)
p=Person('Bill')
print(p)
```

另一个例子:

```python
class Employee:

   empCount = 0



   '所有员工的基类'

   def __init__(self, name, salary):

​        self.name = name

​        self.salary = salary

​        Employee.empCount += 1

   

   def displayCount(self):

​        print ("Total Employee %d" % Employee.empCount)



   def displayEmployee(self):

​        print ("Name :", self.name) 

​        print ("Salary :", self.salary)

"创建 Employee 类的第一个对象"

emp1 = Employee("Zara", 2000)

"创建 Employee 类的第二个对象"

emp2 = Employee("Manni", 5000)



emp1.displayEmployee()# 这里注意:如果想要调用类里面的方法,要记得在方法名字后面加"()"

emp2.displayEmployee()

print ("Total Employee %d" % Employee.empCount)
```



#### 2. Dunder or magic methods <u>__</u>setattr<u>__</u>()

**1、实例属性管理__dict__**

下面的测试代码中定义了三个实例属性，每个实例属性注册后都print()此时的__dict__，代码如下：

```python
class AnotherFun:
    def __init__(self):
        self.name = "Liu"
        print(self.__dict__)
        self.age = 12
        print(self.__dict__)
        self.male = True
        print(self.__dict__)
another_fun = AnotherFun()
```



得到的结果显示出，每次实例属性赋值时，都会将属性名和对应值存储到__dict__字典中：

```python3
{'name': 'Liu'}
{'name': 'Liu', 'age': 12}
{'name': 'Liu', 'age': 12, 'male': True}
```



**2、__setattr__()与__dict__**

由于每次类实例进行属性赋值时都会调用__setattr__()，所以可以重载__setattr__()方法，来动态的观察每次实例属性赋值时__dict__()的变化。下面的Fun类重载了__setattr__()方法，并且将实例的属性和属性值作为__dict__的键-值对：

```python
class Fun:
    def __init__(self):
        self.name = "Liu"
        self.age = 12
        self.male = True
        
    def __setattr__(self, key, value):
        print("*"*50)
        print("setting:{},  with:{}".format(key[], value))
        print("current __dict__ : {}".format(self.__dict__))
        # 属性注册
        self.__dict__[key] = value
fun = Fun()    
```



通过在__setattr__()中将属性名作为key，并将属性值作为value，添加到了__dict__中，得到的结果如下：

```python3
**************************************************
setting:name,  with:Liu
current __dict__ : {}
**************************************************
setting:age,  with:12
current __dict__ : {'name': 'Liu'}
**************************************************
setting:male,  with:True
current __dict__ : {'name': 'Liu', 'age': 12}
```

可以看出，__init__()中三个属性赋值时，每次都会调用一次__setattr__()函数。



**3、重载__setattr__()必须谨慎**

由于__setattr__()负责在__dict__中对属性进行注册，所以自己在重载时必须进行属性注册过程，下面是__setattr__()不进行属性注册的例子：

```python
class NotFun:
    def __init__(self):
        self.name = "Liu"
        self.age = 12
        self.male = True
    
    def __setattr__(self, key, value):
        pass
not_fun = NotFun()
print(not_fun.name)
```

由于__setattr__中并没有将属性注册到__dict__中，所以not_fun对象并没有name属性，因此最后的print（not_fun.name）会报出属性不存在的错误：

```python3
AttributeError                            Traceback (most recent call last)
<ipython-input-21-6158d7aaef71> in <module>()
      8         pass
      9 not_fun = NotFun()
---> 10 print(not_fun.name)

AttributeError: 'NotFun' object has no attribute 'name'
```

所以，重载__setattr__时必须要考虑是否在__dict__中进行属性注册。

#### 3.继承(inheritance[ɪnˈherɪtəns])

##### 什么是继承？

继承是一种创建新的类的方式，新创建的叫子类，继承的叫父类、超类、基类。

特点：**子类可以使用父类的属性（特征、技能）**

继承是类与类之间的关系

##### 为什么要继承？

***减少代码冗余、提高重用性***

##### 如何用继承？

单继承

```python
class grandFather():
  	print('我是爷爷')

class Parent(grandFather):
  	print('我是父类')
  
class SubClass(Parent):
	  print('我是子类')
    
sub = SubClass() 

#结果：我是爷爷
#			我是父类
#			我是子类
#注意：类在定义的时候就执行类体代码，执行顺序是从上到下
```

多继承

```python
class Parent2():
    print('我是第二个爹')

class Parent():
    print('我是第一个爹')
    
class SubClass(Parent, Parent2):
    print('我是子类')
    
#	
# 结果：我是第二个爹
#			 我是第一个爹
# 		 我是子类
#注意：类在定义的时候就执行类体代码，执行顺序是从上到下
```

- 使用__bases__方法可以获取子类继承的类

  ```python
  class Parent2():
      print('我是第二个爹')
  
  class Parent():
      print('我是第一个爹')
      
  class SubClass(Parent, Parent2):
      print('我是子类')
  
  print(SubClass.__bases__)
  #注意，如果sub = SubClass(),sub是没有__bases__方法的
  ```

##### 新式类、经典类

- 继承了object的类以及该类的子类，都是新式类。

  在Python3中如果一个类没有继承任何类，则默认继承object类。因此python3中都是新式类

- 没有继承object的类以及该类的子类，都是经典类。

  在Python2中如果一个类没有继承任何类，不会继承object类。因此，只有Python2中有经典类。

##### 继承与抽象

抽象：通过抽象可以得到类，抽象是一种分析的过程。例如：从小猪佩奇、猪八戒、猪刚鬣、猪猪侠这些具体的对象中，可以分析一下，抽象出一个类，这个类就是猪类。接着，可以从猪、猫、狗等中，可以抽象出一个动物类。先分析、抽象之后，就可以通过继承，在程序上实现这个结构。

```python
class Animals:
  	pass
  
class Pig(Animals):
  	pass
 
class Dog(Animals):
  	pass
  
class Cat(Animals):
  	pass
```

##### 派生类

概念：派生，就是在子类继承父类的属性的基础上，派生出自己的属性。子类有不同于父类的属性，这个子类叫做派生类。通常情况下，子类和派生类是同一个概念，因为子类都是有不同于父类的属性，如果子类和父类属性相同，就没必要创建子类了。

```python
class Animals:
		pass
		
class Dog(Animals):
  	pass
  
#这时候Dog类不是派生类
class Animals:
  	def __init__(self, name):
      	self.name = name
       
    def walk(self):
				print('我会走')
       
class Dog(Animals):
  	#Dog类派生出bite功能
    #派生：狗有咬人的技能
  	def bite(self):
      	print('我会咬人')
```

##### 组合

除了继承之外，还有一种提高重用性的方式：组合

组合指的是，在一个类A中，使用另一个类B的对象作为类A的数据属性（特征）（变量），成为类的组合。

```python
#例子：人和手机，人想要有打电话的功能，想要打电话，就需要用到手机，人想要用到手机里面的打电话功能，肯定不能用继承，人继承手机就非常尴尬了，这时候就可以用到组合。
class Mobile():
  	def __init__(self, color):
      	self.color = color
        
    def call(self):
      	print('老子可以打电话')
        
class People():
  	def __init__(self, name, mobile):
      	self.name = name
        self.mobile = mobile
        
mobile = Mobile('yellow')
people = People('小白', mobile)
people.mobile.call()

#结果：老子可以打电话
```

继承建立了派生类和基类的关系，是一种是的关系，比如白马是马，人是动物。

组合建立了两个类之间'有'的关系，比如人有手机，然后人可以使用手机打电话。

##### 属性查找顺序

对象查找属性的顺序：对象自己的 - > 所在类中 -> 找父类 - >父类的父类 ->Object



#### 4. super() 重写(overrides)和继承(inheritance)

今天我们介绍的主角是super(), 在类的继承里面super()非常常用， 它解决了子类调用父类方法的一些问题， 父类多次被调用时只执行一次， 优化了执行逻辑，下面我们就来详细看一下。

举一个例子：



```python
class Foo:
  def bar(self, message):
    print(message)
```



```ruby
>>> Foo().bar("Hello, Python.")
Hello, Python.
```

当存在继承关系的时候，有时候需要在子类中调用父类的方法，此时最简单的方法是把对象调用转换成类调用，需要注意的是这时self参数需要显式传递，例如：



```ruby
class FooParent:
  def bar(self, message):
    print(message)
class FooChild(FooParent):
  def bar(self, message):
    FooParent.bar(self, message)
```



```ruby
>>> FooChild().bar("Hello, Python.")
Hello, Python.
```

这样做有一些缺点，比如说如果修改了父类名称，那么在子类中会涉及多处修改，另外，Python是允许多继承的语言，如上所示的方法在多继承时就需要重复写多次，显得累赘。为了解决这些问题，Python引入了super()机制，例子代码如下：



```ruby
class FooParent:
  def bar(self, message):
    print(message)
class FooChild(FooParent):
  def bar(self, message):
    super(FooChild, self).bar(message)
```



```ruby
>>> FooChild().bar("Hello, Python.")
Hello, Python.
```

表面上看 super(FooChild, self).bar(message)方法和FooParent.bar(self, message)方法的结果是一致的，实际上这两种方法的内部处理机制大大不同，当涉及多继承情况时，就会表现出明显的差异来，直接给例子：

代码一：



```python
class A:
  def __init__(self):
    print("Enter A")
    print("Leave A")
class B(A):
  def __init__(self):
    print("Enter B")
    A.__init__(self)
    print("Leave B")
class C(A):
  def __init__(self):
    print("Enter C")
    A.__init__(self)
    print("Leave C")
class D(A):
  def __init__(self):
    print("Enter D")
    A.__init__(self)
    print("Leave D")
class E(B, C, D):
  def __init__(self):
    print("Enter E")
    B.__init__(self)
    C.__init__(self)
    D.__init__(self)
    print("Leave E")
E()
```

结果：



```undefined
Enter E
Enter B
Enter A
Leave A
Leave B
Enter C
Enter A
Leave A
Leave C
Enter D
Enter A
Leave A
Leave D
Leave E
```

执行顺序很好理解，唯一需要注意的是公共父类A被执行了多次。

代码二：



```python
class A:
  def __init__(self):
    print("Enter A")
    print("Leave A")
class B(A):
  def __init__(self):
    print("Enter B")
    super(B, self).__init__()
    print("Leave B")
class C(A):
  def __init__(self):
    print("Enter C")
    super(C, self).__init__()
    print("Leave C")
class D(A):
  def __init__(self):
    print("Enter D")
    super(D, self).__init__()
    print("Leave D")
class E(B, C, D):
  def __init__(self):
    print("Enter E")
    super(E, self).__init__()
    print("Leave E")
E()
```

结果：



```undefined
Enter E
Enter B
Enter C
Enter D
Enter A
Leave A
Leave D
Leave C
Leave B
Leave E
```

在super机制里可以保证公共父类仅被**执行一次**，至于执行的顺序，是按照MRO（Method Resolution Order）：方法解析顺序 进行的。后续会详细介绍一下这个MRO机制。



#### 5. %s 字符串, %d 整型, %f 浮点型(%操作符的使用)

##### %s 字符串

```python
string="hello"  

#%s打印时结果是hello  
print "string=%s" % string      # output: string=hello  

#%2s意思是字符串长度为2，当原字符串的长度超过2时，按原长度打印，所以%2s的打印结果还是hello  
print "string=%2s" % string     # output: string=hello  

#%7s意思是字符串长度为7，当原字符串的长度小于7时，在原字符串左侧补空格，  
#所以%7s的打印结果是  hello  
print "string=%7s" % string     # output: string=  hello  

#%-7s意思是字符串长度为7，当原字符串的长度小于7时，在原字符串右侧补空格，  
#所以%-7s的打印结果是  hello  
print "string=%-7s!" % string     # output: string=hello  !  

#%.2s意思是截取字符串的前2个字符，所以%.2s的打印结果是he  
print "string=%.2s" % string    # output: string=he  

#%.7s意思是截取字符串的前7个字符，当原字符串长度小于7时，即是字符串本身，  
#所以%.7s的打印结果是hello  
print "string=%.7s" % string    # output: string=hello  

#%a.bs这种格式是上面两种格式的综合，首先根据小数点后面的数b截取字符串，  
#当截取的字符串长度小于a时，还需要在其左侧补空格  
print "string=%7.2s" % string   # output: string=     he  
print "string=%2.7s" % string   # output: string=hello  
print "string=%10.7s" % string  # output: string=     hello  

#还可以用%*.*s来表示精度，两个*的值分别在后面小括号的前两位数值指定  
print "string=%*.*s" % (7,2,string)      # output: string=     he  
```



##### %d 整型

```python
num=14  
  
#%d打印时结果是14  
print "num=%d" % num            # output: num=14  
  
#%1d意思是打印结果为1位整数，当整数的位数超过1位时，按整数原值打印，所以%1d的打印结果还是14  
print "num=%1d" % num           # output: num=14  
  
#%3d意思是打印结果为3位整数，当整数的位数不够3位时，在整数左侧补空格，所以%3d的打印结果是 14  
print "num=%3d" % num           # output: num= 14  
  
#%-3d意思是打印结果为3位整数，当整数的位数不够3位时，在整数右侧补空格，所以%3d的打印结果是14_  
print "num=%-3d" % num          # output: num=14_  
  
#%05d意思是打印结果为5位整数，当整数的位数不够5位时，在整数左侧补0，所以%05d的打印结果是00014  
print "num=%05d" % num          # output: num=00014  
  
#%.3d小数点后面的3意思是打印结果为3位整数，  
#当整数的位数不够3位时，在整数左侧补0，所以%.3d的打印结果是014  
print "num=%.3d" % num          # output: num=014  
  
#%.0003d小数点后面的0003和3一样，都表示3，意思是打印结果为3位整数，  
#当整数的位数不够3位时，在整数左侧补0，所以%.3d的打印结果还是014  
print "num=%.0003d" % num       # output: num=014  
  
#%5.3d是两种补齐方式的综合，当整数的位数不够3时，先在左侧补0，还是不够5位时，再在左侧补空格，  
#规则就是补0优先，最终的长度选数值较大的那个，所以%5.3d的打印结果还是  014  
print "num=%5.3d" % num         # output: num=  014  
  
#%05.3d是两种补齐方式的综合，当整数的位数不够3时，先在左侧补0，还是不够5位时，  
#由于是05，再在左侧补0，最终的长度选数值较大的那个，所以%05.3d的打印结果还是00014  
print "num=%05.3d" % num        # output: num=00014  
  
#还可以用%*.*d来表示精度，两个*的值分别在后面小括号的前两位数值指定  
#如下，不过这种方式04就失去补0的功能，只能补空格，只有小数点后面的3才能补0  
print "num=%*.*d" % (04,3,num)  # output: num= 014  
```



##### %f 浮点型

```python
import math  

#%a.bf，a表示浮点数的打印长度，b表示浮点数小数点后面的精度  

#只是%f时表示原值，默认是小数点后5位数  
print "PI=%f" % math.pi             # output: PI=3.141593  

#只是%9f时，表示打印长度9位数，小数点也占一位，不够左侧补空格  
print "PI=%9f" % math.pi            # output: PI=_3.141593  

#只有.没有后面的数字时，表示去掉小数输出整数，03表示不够3位数左侧补0  
print "PI=%03.f" % math.pi          # output: PI=003  

#%6.3f表示小数点后面精确到3位，总长度6位数，包括小数点，不够左侧补空格  
print "PI=%6.3f" % math.pi          # output: PI=_3.142  

#%-6.3f表示小数点后面精确到3位，总长度6位数，包括小数点，不够右侧补空格  
print "PI=%-6.3f" % math.pi         # output: PI=3.142_  

#还可以用%*.*f来表示精度，两个*的值分别在后面小括号的前两位数值指定  
#如下，不过这种方式06就失去补0的功能，只能补空格  
print "PI=%*.*f" % (06,3,math.pi)   # output: PI=_3.142  
```



#### 6. split()

split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
os.path.split()：按照路径将文件名和路径分割开

```python
import h5py

dir_order_test = "data/test_order.h5"

with h5py.File(dir_order_test, "r") as hf:

  test_order = hf["order"][:]

print("This is test_order:",test_order)

print("This is length of test_order: ",len(test_order))



f = open("data/Annotations.txt", "r") # There are 4143 lines.

dataset = f.readlines()

print("The dataset contains %d samples" % (len(dataset)))

f.close()



data = dataset[test_order[0]]

x = data.split("&")

print("This is x1:", x[1])
```



#### 7. os.listdir() 方法

##### 语法

**listdir()**方法语法格式如下：

```
os.listdir(path)
```

##### 参数

- **path** -- 需要列出的目录路径

##### 返回值

返回指定路径下的文件和文件夹列表。



```python
import os
raw_video_dir = "data/AVE"  # videos in AVE dataset
lis = os.listdir(raw_video_dir) # 返回指定路径下的文件和文件夹列表。
print("This is lis: ",lis)
```

This output:

```powershell
['---1_cCGK4M.mp4', '--12UOziMF0.mp4', '--5zANFBYzQ.mp4', '--9O4XZOge4.mp4', '--bSurT-1Ak.mp4', '--d2Z5qR4qQ.mp4', '--euLrzIU2Q.mp4', '--fG9gtFqJ0.mp4'......
```



#### 8. len() 方法

##### 描述

len() 方法返回列表元素个数。

##### 语法

len()方法语法：

```
len(list)
```

##### 参数

- list -- 要计算元素个数的列表。

##### 返回值

返回列表元素个数。

##### 实例

以下实例展示了 len()函数的使用方法：

```python
#!/usr/bin/python

list1, list2 = [123, 'xyz', 'zara'], [456, 'abc']

print "First list length : ", len(list1);
print "Second list length : ", len(list2);
```

以上实例输出结果如下：



#### 9. random.choice()

**choice()** 方法返回一个列表，元组或字符串的随机项。

```python
random.choice( seq  )
```

- seq -- 可以是一个列表，元组或字符串。

```python
import random

print ("choice([1, 2, 3, 5, 9]) : ", random.choice([1, 2, 3, 5, 9]))
print ("choice('A String') : ", random.choice('A String'))
```



output:

```python
choice([1, 2, 3, 5, 9]) :  2
choice('A String') :  n
```



#### 10. Python中的for in if 用法

1.if in 判断

```python
def demo():
    L = ["1", "2", "3"]
    if "1" or "4" in L:  # 注意if in组合居于的结尾行有以一个冒号
        print("第一个条件成立")
    if "1" in L and "4" in L:
        print("第二个条件成立")
```

输出结果：

```powershell
第一个条件成立
```


2. for in 循环 

```python
def demo():
    for i in [1, 2, 3]:
        print(i)
```

 输出结果：

```powershell
1
2
3
```

3.for in range()函数

```python
def demo():
    for i in range(3):
        print(i)
```

 输出结果：

```powershell
0
1
2
```

 4.简单的for in if

```python
def demo():
    a = [12, 3, 4, 6, 7, 13, 21]
    newList1 = [x for x in a]
    print(newList1)
    newList2 = [x for x in a if x % 2 == 0]
    print(newList2)
```

输出结果：

```powershell
[12, 3, 4, 6, 7, 13, 21]
[12, 4, 6]
```

newList1构建了一个与a具有相同元素的List，newList2是从a中选取满足x%2==0的元素组成的List

5.嵌套的for in if

```python
def demo():
    a = [12, 3, 4, 7]
    b = ['a', 'x']
    newList1 = [(x, y) for x in a for y in b]
    print(newList1)
    newList2 = [(x, y) for x in a for y in b if x % 2 == 0]
    print(newList2)
```

输出结果：

```powershell
[(12, 'a'), (12, 'x'), (3, 'a'), (3, 'x'), (4, 'a'), (4, 'x'), (7, 'a'), (7, 'x')]
[(12, 'a'), (12, 'x'), (4, 'a'), (4, 'x')]
```



#### 11. float("inf")

表示正负无穷：

```python
# 正无穷 

print(float("inf")) 

print(float("inf")+1) 

# 负无穷 

print(float("-inf")) 

print(float("-inf")+1)
```



output:

```powershell
inf 

inf 

-inf 

-inf
```



```python
if float(1/3)>float("inf"):
    print(0)
else:
    print(1)
if float(1/3)>float("-inf"):
    print(0)
else:
    print(1)
```

output:

```powershell
1 

0
```



#### 12.try ... excapt ... finally

在编码中难免会遇到各种各样的问题，尤其是在对数据进行处理的时候会因为数据的各种问题而抛出异常，如果将数据舍弃太可以，所以数据都过一遍逻辑又太费时间。如果只是对出错的部分进行处理的话会很好的解决问题。

Python中错误处理的语句是：

try....except.....finally

在有可能出错的代码前面加上try，然后捕获到错误之后，在except下处理，finally部分无论try会不是捕获错误都会执行，而且不是必须的。

以简单的除以0的为例：


```python
i = 0

j = 12

try:

​    n = j/i

except ZeroDivisionError as e:

​    print("except:",e)

​    i = 1

​    n = j/i

except ValueError as value_err:  #可以写多个捕获异常

​    print("ValueError")

finally:

​    print("final print")
```

output:

```powershell
except: division by zero 

final print
```



#### 13.Python—print(f “{}”) 

python的print字符串前面加f表示格式化字符串，加f后可以在字符串里面使用用花括号括起来的变量和表达式，如果字符串里面没有表达式，那么前面加不加f输出应该都一样.

```python
a=1

b=2

S=a+b

P=a*b

print(f"Sum of a and b is {S}, and product is {P}")


```

output:

```powershell
Sum of a and b is 3, and product is 2
```



#### 14.eval()

eval函数就是将str to list、dict、tuple.

1.str to list

```python
a = "[[1,2], [3,4], [5,6], [7,8], [9,0]]"

print(type(a))

b = eval(a)

print(type(b))

print(b)
```

output:

```bash
<class 'str'> 

<class 'list'> 

[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]
```

2.str to dic

```python
a = "{1: 'a', 2: 'b'}"

print(type(a))

b = eval(a)

print(type(b))

print(b)
```

output:

```bash
<class 'str'> 

<class 'dict'> 

{1: 'a', 2: 'b'}
```

3.str to tuple

```python
a = "([1,2], [3,4], [5,6], [7,8], (9,0))"

print(type(a))

b=eval(a)

print(type(b))

print(b)
```

output:

```python
<class 'str'> 

<class 'tuple'> 

([1, 2], [3, 4], [5, 6], [7, 8], (9, 0))
```



#### 15.str.zfill(width)

Python zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0。

```python
str = "this is string example....wow!!!";

print str.zfill(40);
print str.zfill(50);
```

this output:

```bash
00000000this is string example....wow!!!
000000000000000000this is string example....wow!!!
```



#### 16. About [:,integer]

```python
import numpy as np

y_predprob =np.array([[0.9,0.1],[0.6,0.4],[0.65,0.35],[0.2,0.8]])
y_scores=y_predprob[:,1] 
print(f"This is y_scroes:{y_scores}") 
x_scores=y_predprob[1,:] 
x_scores
```

output:

```bash
This is y_scroes:[0.1  0.4  0.35 0.8 ]

array([0.6, 0.4])
```



#### 17.About python 中单引号和双引号区别

在Python中使用单引号或双引号是没有区别的，都可以用来表示一个[字符串](https://so.csdn.net/so/search?q=字符串&spm=1001.2101.3001.7020)。但是这两种通用的表达方式可以避免出错之外，还可以减少转义字符的使用，使程序看起来更清晰。

举两个例子：
1、包含单引号的字符串
定义一个字符串my_str，其值为： I’m a student，可以用转义字符和不用转义字符\

```python
my_str = 'I\ 'm a student'
my_str = "I'm a student"
```

2、包含双引号的字符串
定义一个字符串my_str，其值为： Jason said “I like you” ，可以用转义字符和不用转义字符\

```python
my_str = "Jason said \"I like you\""
my_str = 'Jason said "I like you"'
```



#### 18.删减字符串其中的内容: string.replace() 函数



```python
string="./data/short-audio-master/short_audio/0_1-100038-A-14.wav"

ret=string.replace("./","")# 这里将'./'转化为空.

print(f"This is the answer: {ret}")
```

output:

```bash
This is the answer: data/short-audio-master/short_audio/0_1-100038-A-14.wav
```



## About opencv4

#### Regular Contour detection

##### cv2.threshold()函数

```python
Python: cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
```

在其中：

src：表示的是图片源
thresh：表示的是阈值（起始值）
maxval：表示的是最大值
type：表示的是这里划分的时候使用的是什么类型的算法**，常用值为0（cv2.THRESH_BINARY）**

```python
- cv2.THRESH_BINARY //大于ret的pixels变成255,反之则to zero
- cv2.THRESH_BINARY_INV //和第一个相反
- cv2.THRESH_TRUNC //大于thresh，变成thresh,反之的不变 "The maxValue is ignored"
- cv2.THRESH_TOZERO // 大于thresh,不变。反之to zero
- cv2.THRESH_TOZERO_INV //大于thresh，to zero.反之不变。
```

##### cv2.drawContours()

```python
cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
```

**第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓**

**第二个参数是轮廓本身，在Python中是一个list;**

**第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。**

steps:详情可以参考下面的大致代码.

1. convert img(This is image is gray usually) to binary image(thresh)
2.  put binary image(thresh) into findContours(), and then get return value(contour)
3. convert img to color via cvtColor().
4. put contour and color into drawContour(), and then get img( it is new one)

##### cv2.waitKey()

是一个键盘绑定函数。它的时间量度是毫秒ms。函数会等待（n）里面的n毫秒，看是否有键盘输入。若有键盘输入，则返回按键的ASCII值。没有键盘输入，则返回-1.一般设置为0，他将无线等待键盘的输入。

##### cv2.destroyAllWindows() 

用来删除窗口的，（）里不指定任何参数，则删除所有窗口，删除特定的窗口，往（）输入特定的窗口值。


```python
import cv2
import numpy as np

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
print(OPENCV_MAJOR_VERSION)
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255

ret, thresh = cv2.threshold(img, 127, 255, 0)
print("This is ret",ret)
cv2.imshow("threshold",thresh)

if OPENCV_MAJOR_VERSION >= 4:
    # OpenCV 4 or a later version is being used.
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)  
    # int型的mode，定义轮廓的检索模式： CV_RETR_TREE "检测所有轮廓"
    # int型的method，定义轮廓的近似方法： cv2.CHAIN_APPROX_SIMPLE  "仅保存轮廓的拐点信息"  
    # 这里的第一个参数thresh必须是binary图片（即非黑即白）                
else:
    # OpenCV 3 or an earlier version is being used.
    # cv2.findContours has an extra return value.
    # The extra return value is the thresholded image, which is
    # unchanged, so we can ignore it.
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # gray to BGR image 
cv2.imshow("color",color)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2)# 第一个参数必须是三通道图片
cv2.imshow("contours", color)
cv2.imshow("img",img)
cv2.waitKey() # 让函数图像始终显示,否则只会显示一下就闪退。等待keyboard(不是鼠标)输入然后显示下一行.
cv2.destroyAllWindows() # delete all windows
```



#### irregular contour detection

##### Gauss pyramid [ˈpɪrəmɪd] 

 操作一次一个 MxN 的图像就变成了一个 M/2xN/2 的图像。所以这幅图像的面积就变为原来图像面积的四分之一,这被称为 Octave。连续进行这样的操作我们就会得到一个分辨率不断下降的图像金字塔。使用函数 
cv2.pyrDown() 和 cv2.pyrUp() 构建图像金字塔。

##### cv2.IMREAD_UNCHANGED：

顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代

PS：alpha通道，又称A通道，是一个8位的灰度通道，该通道用256级灰度来记录图像中的透明度复信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明.

##### cv2.pyrDown()

cv2.pyrDown() 从一个高分辨率大尺寸的图像向上构建一个金字塔（尺寸变小，分辨率降低）

```python
cv2.pyrDown(src, dst=None, dstsize=None, borderType=None)
```

函数的作用：
对图像进行滤波然后进行下采样

参数含义：
src：表示输入图像
dst：表示输出图像
dstsize：表示输出图像的大小
borderType：表示图像边界的处理方式

```python
import cv2

import numpy as np



img0 = cv2.imread("hammer.jpg")

img1 = cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED)

img  = cv2.pyrDown(cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED))

cv2.imshow("img0",img0)

cv2.imshow("img1",img1)

cv2.imshow("img",img)



cv2.waitKey()

cv2.destroyAllWindows()
```



##### cv2.boundingRect（）和cv2.rectangle（）

cv2.boundingRect(img) 这个函数可以获得一个图像的最小矩形边框一些信息，参数img是一个轮廓点集合，也就是它的参数，可以通过cv2.findContours获取.它可以返回四个参数，左上角坐标，矩形的宽高，一般形式为：

```python
x,y,w,h = cv2.boundingRect(img)
```

配合cv2.rectangle（）可以画出该最小边框，cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)这个函数功能就是画框的函数。

```python
 cv2.rectangle(image, start_point, end_point, color, thickness)
```

##### cv2.minAreaRect(), cv2.boxPoints(), np.int0()

OpenCV provides a function cv2.minAreaRect() for finding the minimum area rotated rectangle. This takes as input a 2D point set and returns a Box2D structure which contains the following details – (center(x, y), (width, height), angle of rotation). The syntax is given below.

```python
(center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
```

But to draw a rectangle, we need 4 corners of the rectangle. So, to convert the Box2D structure to 4 corner points, OpenCV provides another function cv2.boxPoints(). This takes as input the Box2D structure and returns the 4 corner points. The 4 corner points are ordered clockwise starting from the point with the highest y. The syntax is given below.

```python
points = cv2.boxPoints(box)
```

Before drawing the rectangle, you need to convert the 4 corners to the integer type. You can use np.int32 or np.int64 (Don’t use np.int8 as it permits value up to 127 and leads to truncation after that). Sometimes, you might see np.int0 used, don’t get confused, this is equivalent to np.int32 or np.int64 depending upon your system architecture. The full code is given below.

```
rect=cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
```

This is output of np.int0 method:

```powershell
This is cooradiates of box: [[ 31.05075  495.5042  ] 

[ 96.635895  65.38774 ] 

[481.40945  124.0589  ] 

[415.8243   554.17535 ]] 

after inte0(box) method: [[ 31 495] 

[ 96  65] 

[481 124] 

[415 554]]
```

##### cv2.minEnclosingCircle()





```python
import cv2

import numpy as np



img = cv2.pyrDown(cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED))



ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)



contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



for c in contours:

​    \# find bounding box coordinates

​    x, y, w, h = cv2.boundingRect(c)

​    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)



​    \# find minimum area

​    rect = cv2.minAreaRect(c)

​    \# calculate coordinates of the minimum area rectangle

​    box = cv2.boxPoints(rect)

​    \# normalize coordinates to integers

​    box = np.int0(box)

​    \# draw contours

​    cv2.drawContours(img, [box], 0, (0,0, 255), 3)



​    \# calculate center and radius of minimum enclosing circle

​    (x, y), radius = cv2.minEnclosingCircle(c)

​    \# cast to integers

​    center = (int(x), int(y))

​    radius = int(radius)

​    \# draw the circle

​    img = cv2.circle(img, center, radius, (0, 255, 0), 2)



cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

cv2.imshow("contours", img)



cv2.waitKey()

cv2.destroyAllWindows()
```



#### Contours Hierarchy

In this image, there are a few shapes which I have numbered from **0-5**. *2 and 2a* denotes the external and internal contours of the outermost box.

![hierarchy](/home/jiang/桌面/About Python and some image algorithm/pictures source/hierarchy.png)

In this image, there are a few shapes which I have numbered from **0-5**. *2 and 2a* denotes the external and internal contours of the outermost box.

Here, contours 0,1,2 are **external or outermost**. We can say, they are in **hierarchy-0** or simply they are in **same hierarchy level**.

Next comes **contour-2a**. It can be considered as a **child of contour-2** (or in opposite way, contour-2 is parent of contour-2a). So let it be in **hierarchy-1**. Similarly contour-3 is child of contour-2 and it comes in next hierarchy. Finally contours 4,5 are the children of contour-3a, and they come in the last hierarchy level. From the way I numbered the boxes, I would say contour-4 is the first child of contour-3a (It can be contour-5 also).

I mentioned these things to understand terms like **same hierarchy level**, **external contour**, **child contour**, **parent contour**, **first child** etc. Now let's get into OpenCV.



##### Hierarchy Representation in OpenCV

So each contour has its own information regarding what hierarchy it is, who is its child, who is its parent etc. OpenCV represents it as an array of four values : **[Next, Previous, First_Child, Parent]**

*"Next denotes next contour at the same hierarchical level."*

For eg, take contour-0 in our picture. Who is next contour in its same level ? It is contour-1. So simply put Next = 1. Similarly for Contour-1, next is contour-2. So Next = 2.

What about contour-2? There is no next contour in the same level. So simply, put Next = -1. What about contour-4? It is in same level with contour-5. So its next contour is contour-5, so Next = 5.

*"Previous denotes previous contour at the same hierarchical level."*

It is same as above. Previous contour of contour-1 is contour-0 in the same level. Similarly for contour-2, it is contour-1. And for contour-0, there is no previous, so put it as -1.

*"First_Child denotes its first child contour."*

There is no need of any explanation. For contour-2, child is contour-2a. So it gets the corresponding index value of contour-2a. What about contour-3a? It has two children. But we take only first child. And it is contour-4. So First_Child = 4 for contour-3a.

*"Parent denotes index of its parent contour."*

It is just opposite of **First_Child**. Both for contour-4 and contour-5, parent contour is contour-3a. For contour-3a, it is contour-3 and so on.

- Note

  If there is no child or parent, that field is taken as -1

So now we know about the hierarchy style used in OpenCV, we can check into Contour Retrieval Modes in OpenCV with the help of same image given above. ie what do flags like [cv.RETR_LIST](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71a48b9c2cb1056f775ae50bb68288b875e), [cv.RETR_TREE](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71ab10df56aed56c89a026580adc9431f58), [cv.RETR_CCOMP](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71a7d1d4b509fb2a9a8dc2f960357748752), [cv.RETR_EXTERNAL](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71aa7adc6d6608609fd84650f71b954b981) etc mean?

##### Contour Retrieval Mode

###### 1. RETR_LIST

This is the simplest of the four flags (from explanation point of view). It simply retrieves all the contours, but doesn't create any parent-child relationship. **Parents and kids are equal under this rule, and they are just contours**. ie they all belongs to same hierarchy level.

So here, 3rd and 4th term in hierarchy array is always -1. But obviously, Next and Previous terms will have their corresponding values. Just check it yourself and verify it.

Below is the result I got, and each row is hierarchy details of corresponding contour. For eg, first row corresponds to contour 0. Next contour is contour 1. So Next = 1. There is no previous contour, so Previous = -1. And the remaining two, as told before, it is -1.

```
>>> hierarchy

array([[[ 1, -1, -1, -1],

​        [ 2,  0, -1, -1],

​        [ 3,  1, -1, -1],

​        [ 4,  2, -1, -1],

​        [ 5,  3, -1, -1],

​        [ 6,  4, -1, -1],

​        [ 7,  5, -1, -1],

​        [-1,  6, -1, -1]]])
```

This is the good choice to use in your code, if you are not using any hierarchy features.

###### 2. RETR_EXTERNAL

If you use this flag, it returns only extreme outer flags. All child contours are left behind. **We can say, under this law, Only the eldest in every family is taken care of. It doesn't care about other members of the family :)**.

So, in our image, how many extreme outer contours are there? ie at hierarchy-0 level?. Only 3, ie contours 0,1,2, right? Now try to find the contours using this flag. Here also, values given to each element is same as above. Compare it with above result. Below is what I got :

```
>>> hierarchy

array([[[ 1, -1, -1, -1],

​        [ 2,  0, -1, -1],

​        [-1,  1, -1, -1]]])
```

You can use this flag if you want to extract only the outer contours. It might be useful in some cases.

###### 3. RETR_CCOMP

This flag retrieves all the contours and arranges them to a 2-level hierarchy. ie external contours of the object (ie its boundary) are placed in hierarchy-1. And the contours of holes inside object (if any) is placed in hierarchy-2. If any object inside it, its contour is placed again in hierarchy-1 only. And its hole in hierarchy-2 and so on.

Just consider the image of a "big white zero" on a black background. Outer circle of zero belongs to first hierarchy, and inner circle of zero belongs to second hierarchy.

We can explain it with a simple image. Here I have labelled the order of contours in red color and the hierarchy they belongs to, in green color (either 1 or 2). The order is same as the order OpenCV detects contours.

![ccomp_hierarchy.png](https://docs.opencv.org/4.x/ccomp_hierarchy.png)

image

So consider first contour, ie contour-0. It is hierarchy-1. It has two holes, contours 1&2, and they belong to hierarchy-2. So for contour-0, Next contour in same hierarchy level is contour-3. And there is no previous one. And its first is child is contour-1 in hierarchy-2. It has no parent, because it is in hierarchy-1. So its hierarchy array is [3,-1,1,-1]

Now take contour-1. It is in hierarchy-2. Next one in same hierarchy (under the parenthood of contour-1) is contour-2. No previous one. No child, but parent is contour-0. So array is [2,-1,-1,0].

Similarly contour-2 : It is in hierarchy-2. There is not next contour in same hierarchy under contour-0. So no Next. Previous is contour-1. No child, parent is contour-0. So array is [-1,1,-1,0].

Contour - 3 : Next in hierarchy-1 is contour-5. Previous is contour-0. Child is contour-4 and no parent. So array is [5,0,4,-1].

Contour - 4 : It is in hierarchy 2 under contour-3 and it has no sibling. So no next, no previous, no child, parent is contour-3. So array is [-1,-1,-1,3].

Remaining you can fill up. This is the final answer I got:

```
>>> hierarchy

array([[[ 3, -1,  1, -1],

​        [ 2, -1, -1,  0],

​        [-1,  1, -1,  0],

​        [ 5,  0,  4, -1],

​        [-1, -1, -1,  3],

​        [ 7,  3,  6, -1],

​        [-1, -1, -1,  5],

​        [ 8,  5, -1, -1],

​        [-1,  7, -1, -1]]])

###### 
```

###### 4. RETR_TREE

And this is the final guy, Mr.Perfect. It retrieves all the contours and creates a full family hierarchy list. **It even tells, who is the grandpa, father, son, grandson and even beyond... :)**.

For example, I took above image, rewrite the code for [cv.RETR_TREE](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gga819779b9857cc2f8601e6526a3a5bc71ab10df56aed56c89a026580adc9431f58), reorder the contours as per the result given by OpenCV and analyze it. Again, red letters give the contour number and green letters give the hierarchy order.

![tree_hierarchy.png](https://docs.opencv.org/4.x/tree_hierarchy.png)

image

Take contour-0 : It is in hierarchy-0. Next contour in same hierarchy is contour-7. No previous contours. Child is contour-1. And no parent. So array is [7,-1,1,-1].

Take contour-2 : It is in hierarchy-1. No contour in same level. No previous one. Child is contour-3. Parent is contour-1. So array is [-1,-1,3,1].

And remaining, try yourself. Below is the full answer:

```
>>> hierarchy

array([[[ 7, -1,  1, -1],

​        [-1, -1,  2,  0],

​        [-1, -1,  3,  1],

​        [-1, -1,  4,  2],

​        [-1, -1,  5,  3],

​        [ 6, -1, -1,  4],

​        [-1,  5, -1,  4],

​        [ 8,  0, -1, -1],

​        [-1,  7, -1, -1]]])


```



#### Image moments from opencv

![moment-1](/home/jiang/桌面/About Python and some image algorithm/pictures source/moment-1.jpg)

##### Area:

![moment_area](/home/jiang/桌面/About Python and some image algorithm/pictures source/moment_area.jpg)

For a binary image, this corresponds to counting all the non-zero pixels and that is equivalent to the area. For greyscale image, this corresponds to the sum of pixel intensity values.

##### Centroid:

Centroid simply is the arithmetic mean position of all the points. In terms of image moments, centroid is given by the relation.

![moment_cent](/home/jiang/桌面/About Python and some image algorithm/pictures source/moment_cent.jpg)

This is simple to understand. For instance, for a binary image M10 corresponds to the sum of all non-zero pixels (x-coordinate) and M00 is the total number of non-zero pixels and that is what the centroid is.

Let’s take a simple example to understand how to calculate image moments for a given image.

![mom_cen2](/home/jiang/桌面/About Python and some image algorithm/pictures source/mom_cen2.jpg)

Below are the area and centroid calculation for the above image.

![mom_cen3](/home/jiang/桌面/About Python and some image algorithm/pictures source/mom_cen3.jpg)



#### 1.Convex contours

##### cv2.arcLength

It is also called arc length. It can be found out using cv.arcLength() function. Second argument specify whether shape is a closed contour (if passed True), or just a curve.

```python
perimeter = cv.arcLength(cnt,True)
```



```python
import cv2

import numpy as np



OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])



img = cv2.pyrDown(cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED))



ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),

​                            127, 255, cv2.THRESH_BINARY)



if OPENCV_MAJOR_VERSION >= 4:

​    \# OpenCV 4 or a later version is being used.

​    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,

​                                      cv2.CHAIN_APPROX_SIMPLE)

else:

​    \# OpenCV 3 or an earlier version is being used.

​    \# cv2.findContours has an extra return value.

​    \# The extra return value is the thresholded image, which is

​    \# unchanged, so we can ignore it.

​    _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,

​                                         cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

perimeter = cv2.arcLength(cnt,True)

print("This is a perimeter: ", perimeter)

epsilon = 0.1*cv2.arcLength(cnt,True)

approx = cv2.approxPolyDP(cnt,epsilon,True)

print("This is an approx: ",approx)



black = np.zeros_like(img)

for cnt in contours:

​    epsilon = 0.01 * cv2.arcLength(cnt,True) 

​    approx = cv2.approxPolyDP(cnt,epsilon,True)

​    hull = cv2.convexHull(cnt)

​    cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)

​    cv2.drawContours(black, [approx], -1, (255, 255, 0), 2) 
# contour approximation

​    cv2.drawContours(black, [hull], -1, (0, 0, 255), 2) 
# Convex Hull 



cv2.imshow("hull", black)

cv2.waitKey()

cv2.destroyAllWindows()
```

This is a output:

![hull](/home/jiang/桌面/About Python and some image algorithm/pictures source/hull.png)



#### 2.Hough Detecting Lines

```python
import cv2

import numpy as np



img = cv2.imread('lines.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 120)

minLineLength = 20

maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20,

​                        minLineLength, maxLineGap)

for x1, y1, x2, y2 in lines[0]:

​    cv2.line(img, (x1, y1), (x2, y2), (0,255,0),2)



cv2.imshow("edges", edges)

cv2.imshow("lines", img)

cv2.waitKey()

cv2.destroyAllWindows()
```



#### 3.Hough Detecting Circles

####  

```python


import cv2

import numpy as np



planets = cv2.imread('planet_glow.jpg')

gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)

gray_img = cv2.medianBlur(gray_img, 5)



circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,120,

​                           param1=100,param2=30,minRadius=0,maxRadius=0)



circles = np.uint16(np.around(circles))



for i in circles[0,:]:

​    \# draw the outer circle

​    cv2.circle(planets,(i[0],i[1]),i[2],(0,255,0),2)

​    \# draw the center of the circle

​    cv2.circle(planets,(i[0],i[1]),2,(0,0,255),3)



cv2.imwrite("planets_circles.jpg", planets)

cv2.imshow("HoughCirlces", planets)

cv2.waitKey()

cv2.destroyAllWindows()
```



#### 4.cv2.applyColorMap() 函数



```python
import numpy as np

import cv2



im_gray =  cv2.imread("horses.jpg", cv2.IMREAD_GRAYSCALE)

im_color =  cv2.applyColorMap(im_gray[0:200,0:200], cv2.COLORMAP_JET)

# cv2.COLORMAP_JET 是热力图的模式

cv2.imwrite("try1.jpg", im_color)
```



#### 5.cv2.resize()

interpolation algorithm





## About numpy

#### 1. numpy.zeros_like and numpy.reshape()

Examples:

```python
>>> x = np.arange(6)
>>> x = x.reshape((2, 3))
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.zeros_like(x)
array([[0, 0, 0],
       [0, 0, 0]])
>>> y = np.arange(3, dtype=float)
>>> y
array([0., 1., 2.])
>>> np.zeros_like(y)
array([0.,  0.,  0.])
```

也就是说，先前我们不知道z的shape属性是多少，但是想让z变成只有一列，行数不知道多少，通过`z.reshape(-1,1)`，Numpy自动计算出有12行，新的数组shape属性为(16, 1)，与原来的(4, 4)配套。

```python
z = np.array([[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]])
z.shape
(4, 4)

z.reshape(-1)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

z.reshape(-1, 1)
 array([[ 1],
        [ 2],
        [ 3],
        [ 4],
        [ 5],
        [ 6],
        [ 7],
        [ 8],
        [ 9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16]])
    
# newshape等于-1，列数等于2，行数未知，reshape后的shape等于(8, 2)
z.reshape(-1, 2)
array([[ 1,  2],
       [ 3,  4],
       [ 5,  6],
       [ 7,  8],
       [ 9, 10],
       [11, 12],
       [13, 14],
       [15, 16]])

audio = np.reshape(audio, (1, -1, 1))
    # 这里面维度会变成(1, 441000, 1)

```



#### 2. about numpy.uint8

用opencv处理图像时，可以发现获得的矩阵类型都是uint8

```python
import cv2 as cv
img=cv.imread(hello.png)
print(img)
array([[[...],
        [...],
        [...]]],dtype='uint8')
```

uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
这里要注意如何转化到uint8类型
1: numpy有np.uint8()函数，但是这个函数仅仅是对原数据和0xff相与(和最低1字节数据相与)，这就容易导致如果原数据是大于255的，那么在直接使用np.uint8()后，比第八位更大的数据都被截断了，比如:

```python
a=[2000,100,2]
np.uint8(a)
array([208, 100, 2], dtype=uint8)
```

2: 用cv2.normalize函数配合cv2.NORM_MINMAX，可以设置目标数组的最大值和最小值，然后让原数组等比例的放大或缩小到目标数组，如下面的例子中是将img的所有数字等比例的放大或缩小到0–255范围的数组中，

```python
cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX) 

# 然后改变数据类型 np.array([out],dtype=‘uint8’)
```

总结：
要想将当前的数组作为图像类型来进行各种操作，就要转换到uint8类型，转换的方式推荐使用第二种，因为第一种在值大于255以后就容易丢失。



#### 3.about np.repeat()

```python
import numpy as np

# 随机生成[0,5)之间的数，形状为(1,4),将此数组重复3次

pop = np.random.randint(0, 10, size=(3, 4))
bob = pop.repeat(3, axis=0)
coc = pop.repeat(3, axis=1)
dod = pop.repeat(3, axis=-1)
print("pop\n",pop)
print("bob\n",bob)
print("coc\n",coc)
print("dod\n",dod)
```

 output:

```powershell
pop
 [[3 9 5 8]
 [5 0 1 7]
 [8 4 5 2]]
bob
 [[3 9 5 8]
 [3 9 5 8]
 [3 9 5 8]
 [5 0 1 7]
 [5 0 1 7]
 [5 0 1 7]
 [8 4 5 2]
 [8 4 5 2]
 [8 4 5 2]]
coc
 [[3 3 3 9 9 9 5 5 5 8 8 8]
 [5 5 5 0 0 0 1 1 1 7 7 7]
 [8 8 8 4 4 4 5 5 5 2 2 2]]
dod
 [[3 3 3 9 9 9 5 5 5 8 8 8]
 [5 5 5 0 0 0 1 1 1 7 7 7]
 [8 8 8 4 4 4 5 5 5 2 2 2]]
```



#### PyTorch view和reshape的区别



##### 相同之处

都可以用来重新调整 tensor 的形状。

##### 不同之处

view 函数只能用于 contiguous 后的 tensor 上，也就是只能用于内存中连续存储的 tensor。如果对 tensor 调用过 transpose, permute 等操作的话会使该 tensor 在内存中变得不再连续，此时就不能再调用 view 函数。因此，需要先使用 contiguous 来返回一个 contiguous copy。
reshape 则不需要依赖目标 tensor 是否在内存中是连续的。



#### 4.about numpy.max() and min()

ndarray.max([int axis])

函数功能：求ndarray中指定维度的最大值，默认求所有值的最大值。

axis=0:求各column的最大值

axis=1:求各row的最大值



#### 5. about 简单的归一化函数

```python
import numpy as np

input = np.random.randint(0, 10, size=(10,7,7))

print("This is input: ",input)



def normalize(x, min=0, max=255):


  num, row, col = x.shape

  for i in range(num):

​    xi = x[i, :, :]

​    xi = max * (xi - np.min(xi)) / (np.max(xi) - np.min(xi))

​    x[i, :, :] = xi

  return x

input = normalize(input, 0, 255)


print("This is a new input: ",input)
```



#### 6. about numpy: Set whether to print full or truncated ndarray

If `threshold` is set to infinity `np.inf`, full elements will always be printed without truncation.

```python
np.set_printoptions(threshold=np.inf)
```



#### 7.numpy.argmax()

一维数组: 获取数组最大值的**索引值**

```python
import numpy as np

a = np.array([3, 1, 2, 4, 6, 1])

print(np.argmax(a))
```

output:4

二维数组: 每一行的最大值的**索引值**

```python
import numpy as np

a = np.array([[1, 5, 5, 2],

​              [9, 6, 2, 8],

​              [3, 7, 9, 1]])

print(np.argmax(a, axis=1))
```

output: 

[1 0 2]



## About sklearn

#### 1.about sklearn.preprocessing.MinMaxScaler()



```python
from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

X_train = np.array([[1., -1., 2.],

​          [2., 0., 0.],

​          [0., 1., -1.]])

\# axis=0, 我们可以获取每一列的最小值

print("This is X_train.min(axis=0): ",X_train.min(axis=0))

mini_max_scaler = preprocessing.MinMaxScaler()

X1 = np.array([[1,2,3]]).T

print("This is X1: ",X1)

X1_STD = (X1 - X1.min()) / (X1.max() - X1.min())

print("This is X1.min: ",X1.min())

print("This is X1-min: ",X1 - X1.min())

print("This is X1-10: ",X1 - 10)

print("This is X1_STD: ",X1_STD)

min = 0.1

max = 0.6

X1_scaled = X1_STD * (max - min) + min

print('X1_scaled',X1_scaled)

mini_max_scaler.fit(X_train)

print('min_',mini_max_scaler.min_) # 未知

print('scale_',mini_max_scaler.scale_) # 未知

print('data_max_',mini_max_scaler.data_max_) # 每一列中最大的那个数

print('data_min_',mini_max_scaler.data_min_) # 每一列中最小的那个数

print('data_range_',mini_max_scaler.data_range_) # 每一列的范围
```

output:

```powershell
This is X_train.min(axis=0):  [ 0. -1. -1.] This is X1:  [[1] [2] [3]] This is X1.min:  1 This is X1-min:  [[0] [1] [2]] This is X1-10:  [[-9] [-8] [-7]] This is X1_STD:  [[0. ] [0.5] [1. ]] X1_scaled [[0.1 ] [0.35] [0.6 ]] min_ [0.         0.5        0.33333333] scale_ [0.5        0.5        0.33333333] data_max_ [2. 1. 2.] data_min_ [ 0. -1. -1.] data_range_ [2. 2. 3.]
```



#### 2.sklearn.metrics.roc_auc_score()

```python
sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
```

y_true：样本的真实标签，形状（样本数）
y_score：预测为1的概率值，形状（样本数）

```python
import numpy as np

from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1])

print(y_true.shape) #(4,)

y_predprob =np.array([[0.9,0.1],[0.6,0.4],[0.65,0.35],[0.2,0.8]])

\#print(y_predprob)

y_scores=y_predprob[:,1] #取预测标签为1的概率

print(f"This is y_scroes:{y_scores}") 

auc=roc_auc_score(y_true, y_scores)

print(auc)#0.75
```





## About os

#### 1. os.makedir(path)和os.makedirs(path)

首先说os.mkdir(path)，他的功能是一级一级的创建目录，前提是前面的目录已存在，如果不存在会报异常，比较麻烦，但是存在即有他的道理，当你的目录是根据文件名动态创建的时候，你会发现他虽然繁琐但是很有保障，不会因为你的一时手抖，创建而创建了双层或者多层错误路径，

```python
import os

os.mkdir('d:\hello')    #  正常
os.mkdir('d:\hello\hi') #  正常

#  如果d:\hello目录不存在

#  则os.mkdir('d:\hello\hi')执行失败
```

然后是os.makedirs(path),单从写法上就能猜出他的区别，他可以一次创建多级目录，哪怕中间目录不存在也能正常的（替你）创建，想想都可怕，万一你中间目录写错一个单词.........

```python
import os

os.makedirs('d:\hello')    #  正常
os.makedirs('d:\hello\hi') #  正常

#  如果d:\hello目录不存在

#  则os.makedirs('d:\hello\hi')  #  仍然正常
```



#### 2. os.path.join()

1.如果各组件名首字母不包含'/'，则函数会自动加上

```python
import os

Path1 = 'home'

Path2 = 'develop'

Path3 = 'code'

Path10 = Path1 + Path2 + Path3

Path20 = os.path.join(Path1,Path2,Path3)

print ('Path10 = ',Path10)

print ('Path20 = ',Path20)

```

output:

```powershell
Path10 =  homedevelopcode 

Path20 =  home/develop/code
```

2.如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃

```python
import os 

Path1 = '/home'

Path2 = 'develop'

Path3 = 'code'

Path10 = Path1 + Path2 + Path3

Path20 = os.path.join(Path1, Path2, Path3)

Path30 = os.path.join(Path2, Path1, Path3)

print('Path10 = ',Path10)

print('Path20 = ',Path20)

print('Path30 = ',Path30)
```

output:

```powershell
Path10 =  /homedevelopcode 

Path20 =  /home/develop/code 

Path30 =  /home/code
```

3.如果最后一个组件为空，则生成的路径以一个'/'分隔符结尾

```python
import os

Path1 = 'home'

Path2 = 'develop'

Path3 = ''

Path10 = Path1 + Path2 + Path3

Path20 = os.path.join(Path1, Path2, Path3)

Path30 = os.path.join(Path2, Path1, Path3)

print('Path10 = ',Path10)

print('Path20 = ',Path20)

print('Path20 = ',Path30)
```

output:

```python
Path10 =  homedevelop 

Path20 =  home/develop/ 

Path20 =  develop/home/
```



## About json

1、json.dumps()：将dict数据类型转成str

2、json.loads()：将str数据类型转成dict

3、json.dump()：将dict类型的数据转成str，并写入到json文件中

4、json.load()：从json文件中读取数据（<type ‘dict’>）

#### 1.json.dump() 存储

导入模块json
以写入模式打开这个文件，让json能够将数据写入其中
我们使用函数json.dump()将数字列表存储到文件numbers.json中

indent:具有缩进的效果

```python
import json
numbers = [1, 3, 5, 7, 9]
filename = 'numbers.json'
with open(filename, 'w') as f_json:
    json.dump(numbers, f_json,indent=1)
```



#### 2.json.load() 读取

以读取方式打开这个文件
使用函数json.load()加载存储在numbers.json中的信息，
并将其存储到变量numbers中。最后，我们打印恢复的数字列表，看看它是否与number_writer.py中创建的数字列表相同

```python
with open(filename) as f_json:
    numbers2 = json.load(f_json)
print(numbers2)
```





























## About Keras

#### 1. tf.keras.layers.GlobalAveragePooling2D(    data_format=None, **kwargs )

输入参数：
data_format:
输入是一个字符串。“channels_last”(默认) 或者"channels_first"。
channels_last:代表通道数在最后，输入数据的形式是(batch, height, width, channels)；
channels_first:代表通道数在前面，输入数据的形式是(batch, channels, height, width)；
具体还是根据你在keras使用的图片数据的形式。不需要设置，系统会自动匹配。如果你从未设置过，默认是“channels_last”.

```python
import tensorflow as tf

import torch

a=torch.arange(0,24).view(2, 3, 2,2)

a=a.numpy()



print("This is x: ",a)
# 这里得到a是整型，日后应该用浮点型.

y = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(a)

print(y,y.shape)


```



## About Tensorflow

#### 1.构建网络第一层（输入层）

- 用于构建网络的第一层——**输入层**，该层会告诉网络我们的输入的尺寸是什么，这一点很重要。例如使用Model(input=x,output=y)构建网络，这种构建方式很常见，用途很广，详细[参考文章](https://blog.csdn.net/weixin_44441131/article/details/105905536)

```python
tf.keras.layers.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    ragged=False,
    **kwargs,
)
```



参数说明
shape:元组维数，定义输入层神经元对应数据的形状。比如shape=(32, )和shape=32是等价的，表示输入都为长度为32的向量。

batch_size：声明输入的batch_size大小，定义输入层时不需要声明，会在fit时声明，一般在训练时候用

name：给layers起个名字，在整个神经网络中不能重复出现。如果name=None，程序会自动为该层创建名字。

dtype：数据类型，一般数据类型为tf.float32，计算速度更快

sparse：特定的布尔值，占位符是否为sparse

tensor：可选的现有tensor包装到“Input”层。如果设置该参数，该层将不会创建占位符张量



## About pydud

#### 1.截取音频文件

```python
from pydub import AudioSegment

training_list= ["2400275000","2400910000","2405274000","2409891000","2421209000"
,"2423313000","2424577000","2450751000","2455265000","2464807000"]
for i in range(0,len(training_list)):
    print(i)

    # sourceName = 'training_file/training_list[i]/2405274000'

    # 加载mp3文件

​    input_music = AudioSegment.from_mp3('training_file/'+ training_list[i]+'/'+training_list[i]+ '.mp4.mp3')

    # sound_time = input_music.duration_seconds # 获取音频时间

    # 截取音频前20秒的内容

​    output_music = input_music[0:20000]

    # 保存音频 前面为保存的路径，后面为保存的格式

​    output_music.export('training_file/'+ training_list[i]+'/'+training_list[i] + '.wav', format='wav')
```



## About scipy

#### 1.scipy.special.softmax()

```python
from scipy.special import softmax

np.set_printoptions(precision=5)

x = np.array([[1, 0.5, 0.2, 3],

​              [1,  -1,   7, 3],

​              [2,  12,  13, 3]])

m = softmax(x)

print("无论行列，所有元素使用softmax：",m)

print("无论行列，所有元素经过softmax之后进行求和：",m.sum())

m = softmax(x, axis=0)

print("对每一列进行softmax:",m)

print("对每一列进行求和",m.sum(axis=0))

m = softmax(x, axis=1)

print("对每一行进行softmax:",m)

print("对每一行进行求和",m.sum(axis=1))
```

output:

```powershell
无论行列，所有元素使用softmax： 

[[4.48309e-06 2.71913e-06 2.01438e-06 3.31258e-05] 

[4.48309e-06 6.06720e-07 1.80861e-03 3.31258e-05] 

[1.21863e-05 2.68421e-01 7.29644e-01 3.31258e-05]] 

无论行列，所有元素经过softmax之后进行求和： 1.0000000000000002 

对每一列进行softmax: 

[[2.11942e-01 1.01300e-05 2.75394e-06 3.33333e-01] 

[2.11942e-01 2.26030e-06 2.47262e-03 3.33333e-01]

[5.76117e-01 9.99988e-01 9.97525e-01 3.33333e-01]] 

对每一列进行求和 [1. 1. 1. 1.] 

对每一列进行softmax: 

[[1.05877e-01 6.42177e-02 4.75736e-02 7.82332e-01] 

[2.42746e-03 3.28521e-04 9.79307e-01 1.79366e-02] 

[1.22094e-05 2.68929e-01 7.31025e-01 3.31885e-05]] 

对每一列进行求和 [1. 1. 1.]
```





## About torchaudio

#### 1. torchaudio.load()

Load audio file into torch.Tensor object. Refer to [torchaudio.backend](https://pytorch.org/audio/stable/backend.html#backend) for the detail.

```python
import torch

import torchaudio

import matplotlib.pyplot as plt



filename = "/home/jiang/桌面/pytorchforaudio-main/UrbanSound8K/audio/fold2/4201-3-0-0.wav"

waveform,sample_rate = torchaudio.load(filename)

print("Shape of waveform:{}".format(waveform.size())) #音频大小

print("sample rate of waveform:{}".format(sample_rate))#采样率

plt.figure()

plt.plot(waveform.t().numpy())

plt.show()

```

output:

```powershell
Shape of waveform:torch.Size([2, 10937]) 

sample rate of waveform:44100
```

![waveform](/home/jiang/桌面/About Python and some image algorithm/pictures source/waveform.png)

#### 2.torchaudio.transforms.MelSpectrogram()

Create MelSpectrogram for a raw audio signal.

This is a composition of [`torchaudio.transforms.Spectrogram()`](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.Spectrogram) and and [`torchaudio.transforms.MelScale()`](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.MelScale).

```python
torchaudio.transforms.MelSpectrogram()
```

Parameters:

- **sample_rate** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Sample rate of audio signal. (Default: `16000`)
- **n_fft** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Size of FFT, creates `n_fft // 2 + 1` bins. (Default: `400`)
- **win_length** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*None*](https://docs.python.org/3/library/constants.html#None)*,* *optional*) – Window size. (Default: `n_fft`)
- **hop_length** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*None*](https://docs.python.org/3/library/constants.html#None)*,* *optional*) – Length of hop between STFT windows. (Default: `win_length // 2`)
- **f_min** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Minimum frequency. (Default: `0.`)
- **f_max** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*None*](https://docs.python.org/3/library/constants.html#None)*,* *optional*) – Maximum frequency. (Default: `None`)
- **pad** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Two sided padding of signal. (Default: `0`)
- **n_mels** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Number of mel filterbanks. (Default: `128`)
- **window_fn** (*Callable**[**..**,* *Tensor**]**,* *optional*) – A function to create a window tensor that is applied/multiplied to each frame/window. (Default: `torch.hann_window`)
- **power** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: `2`)
- **normalized** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – Whether to normalize by magnitude after stft. (Default: `False`)
- **wkwargs** (*Dict**[**..**,* *..**] or* [*None*](https://docs.python.org/3/library/constants.html#None)*,* *optional*) – Arguments for window function. (Default: `None`)
- **center** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – whether to pad `waveform` on both sides so that the t*t*-th frame is centered at time t \times \text{hop\_length}*t*×hop_length. (Default: `True`)
- **pad_mode** (*string**,* *optional*) – controls the padding method used when `center` is `True`. (Default: `"reflect"`)
- **onesided** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – controls whether to return half of results to avoid redundancy. (Default: `True`)
- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*None*](https://docs.python.org/3/library/constants.html#None)*,* *optional*) – If ‘slaney’, divide the triangular mel weights by the width of the mel band (area normalization). (Default: `None`)
- **mel_scale** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – Scale to use: `htk` or `slaney`. (Default: `htk`)



#### 3.torchaudio.compliance.kaldi.fbank()

Create a fbank from a raw audio signal. This matches the input/output of Kaldi’s compute-fbank-feats. *总而言之用这个函数我们可以得到filterbank的特征*

- Parameters

  - **waveform** (*Tensor*) – Tensor of audio of size (c, n) where c is in the range [0,2)
  - **blackman_coeff** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Constant coefficient for generalized Blackman window. (Default: `0.42`)
  - **channel** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: `-1`)
  - **dither** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Dithering constant (0.0 means no dither). If you turn this off, you should set the energy_floor option, e.g. to 1.0 or 0.1 (Default: `0.0`)
  - **energy_floor** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Floor on energy (absolute, not relative) in Spectrogram computation. Caution: this floor is applied to the zeroth component, representing the total signal energy. The floor on the individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: `1.0`)
  - **frame_length** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Frame length in milliseconds (Default: `25.0`)
  - **frame_shift** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Frame shift in milliseconds (Default: `10.0`)
  - **high_freq** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (Default: `0.0`)
  - **htk_compat** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If true, put energy last. Warning: not sufficient to get HTK compatible features (need to change other parameters). (Default: `False`)
  - **low_freq** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Low cutoff frequency for mel bins (Default: `20.0`)
  - **min_duration** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Minimum duration of segments to process (in seconds). (Default: `0.0`)
  - **num_mel_bins** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Number of triangular mel-frequency bins (Default: `23`)
  - **preemphasis_coefficient** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Coefficient for use in signal preemphasis (Default: `0.97`)
  - **raw_energy** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If True, compute energy before preemphasis and windowing (Default: `True`)
  - **remove_dc_offset** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – Subtract mean from waveform on each frame (Default: `True`)
  - **round_to_power_of_two** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If True, round window size to power of two by zero-padding input to FFT. (Default: `True`)
  - **sample_frequency** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Waveform data sample frequency (must match the waveform file, if specified there) (Default: `16000.0`)
  - **snip_edges** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If True, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame_length. If False, the number of frames depends only on the frame_shift, and we reflect the data at the ends. (Default: `True`)
  - **subtract_mean** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – Subtract mean of each feature file [CMS]; not recommended to do it this way. (Default: `False`)
  - **use_energy** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – Add an extra dimension with energy to the FBANK output. (Default: `False`)
  - **use_log_fbank** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If true, produce log-filterbank, else produce linear. (Default: `True`)
  - **use_power** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If true, use power, else use magnitude. (Default: `True`)
  - **vtln_high** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – High inflection point in piecewise linear VTLN warping function (if negative, offset from high-mel-freq (Default: `-500.0`)
  - **vtln_low** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Low inflection point in piecewise linear VTLN warping function (Default: `100.0`)
  - **vtln_warp** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – Vtln warp factor (only applicable if vtln_map not specified) (Default: `1.0`)
  - **window_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – Type of window (‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’) (Default: `'povey'`)

- Returns

  A fbank identical to what Kaldi would output. The shape is (m, `num_mel_bins + use_energy`) where m is calculated in _get_strided

- Return type

  Tensor

- Example:



```python
import torchaudio
import matplotlib.pyplot as plt


filename='/home/jiang/桌面/ast-master/egs/short_audio/data/short-audio-master/short_audio/0_1-100038-A-14.wav'

waveform, sample_rate = torchaudio.load(filename)
fbank = torchaudio.compliance.kaldi.fbank(waveform)
print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')
plt.show()
```



## About random

#### 1、python中的random函数

random() 方法返回随机生成的一个实数，它在[0,1)范围内

```python
import random
random.random()

#randint函数，返回指定范围的一个随机整数，包含上下限
random.randint(0,99)#返回0~99之间的整数

#randrange函数，randrange(0,101,2)可以用来选曲0~100之间的偶数
```

#### 2、random.seed(int)

给随机数对象一个种子值，用于产生随机序列。
对于同一个种子值的输入，之后产生的随机数序列也一样。

通常是把时间秒数等变化值作为种子值，达到每次运行产生的随机系列都不一样
seed() 省略参数，意味着使用当前系统时间生成随机数

```python
random.seed(10)
print random.random()   #0.57140259469
random.seed(10)
print random.random()   #0.57140259469  同一个种子值，产生的随机数相同
print random.random()   #0.428889054675

random.seed()           #省略参数，意味着取当前系统时间
print random.random()
random.seed()
print random.random()
```

#### 3、随机正态浮点数random.uniform(u,sigma)

```python
print random.uniform(1,5)
```

#### 4、按步长随机在上下限范围内取一个随机数

```python
#random.randrange(start,stop,step)
print random.randrange(20,100,5)
```



#### 5、随机选择字符

```python
#随机的选取n个字符
print(random.sample('abcdefghijk',3))

#随机的选取一个字符
print(random.choice('af/fse.faek``fs'))

#随机选取几个字符，再拼接成新的字符串
print string.join(random.sample('abcdefhjk',4)).replace(" ","")

```

#### 6、random.shuffle

对list列表随机打乱顺序，也就是洗牌

shuffle只作用于list，对str会报错，比如‘abcdfed’,
而[‘1’,‘2’,‘3’,‘5’,‘6’,‘7’]可以

```python
item1=[1,2,3,4,5,6,7]
print item1
random.shuffle(item1)
print item1

item2=['1','2','3','5','6','7']
print item2
random.shuffle(item2)
print item2

```

#### 7、numpy模块中的randn和rand函数

numpy.random.randn(d0,d1,…,dn),正太随机

numpy.random.rand(d0,d1,…,dn)，选择[0,1]范围内的随机数

```python
import numpy
numpy.random.randn(2,3)
array([[ 1.62434536, -0.61175641, -0.52817175],
       [-1.07296862,  0.86540763, -2.3015387 ]])

numpy.random.rand(2,3)
array([[0.41919451, 0.6852195 , 0.20445225],
       [0.87811744, 0.02738759, 0.67046751]])
```



## About matplotlib

1. plot histogram by matplotlib

```python
import matplotlib.pyplot as plt

import mpl_toolkits.axisartist.axislines as axislines

import numpy as np





listA=[10]*13+[20]*25+[30]*42+[40]*50+[50]*58+[60]*72+[70]*47+[80]*63+[90]*45+[100]*41

\# An "interface" to matplotlib.axes.Axes.hist() method

n, bins, patches = plt.hist(x=listA, bins=10, color='#0504aa',

​                            alpha=0.7, rwidth=0.8, orientation='horizontal',histtype='barstacked')

plt.grid(axis='y', alpha=0.75)

plt.xticks(np.arange(0,110,10))

plt.xlabel('Numbers of intervals')



plt.yticks([0,10,20,30,40,50,60,70,80,90,100,110],['','40ms-140ms','140ms-240ms','240ms-340ms','340ms-440ms','440ms-540ms','540ms-640ms','640ms-740ms','740ms-840ms','840ms-940ms','940ms-inf',''])

plt.ylabel('Sound length interval')



plt.title('Figure 1: Interval histograms of different sound lengths at millisecond level(DatasetA)',y=-0.3)
```





























