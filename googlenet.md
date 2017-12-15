```{.python .input  n=1}
from mxnet.gluon import nn
from mxnet import ndarray as nd

```

```{.python .input  n=77}
class Inception(nn.Block):
    def __init__(self,n1_1,n2_1,n2_3,n3_1,n3_5,n4_1,**kwargs):
        super(Inception, self).__init__(**kwargs)
        with self.name_scope():
            self.path1_conv_1 = nn.Conv2D(n1_1, kernel_size = 1,activation = 'relu')
            
            self.path2_conv_1 = nn.Conv2D(n2_1, kernel_size = 1, activation = 'relu')
            self.path2_conv_3 = nn.Conv2D(n2_3, kernel_size = 3, activation = 'relu',padding = 1)
            
            self.path3_conv_1 = nn.Conv2D(n3_1, kernel_size = 1, activation = 'relu')
            self.path3_conv_5 = nn.Conv2D(n3_5, kernel_size = 5, activation = 'relu',padding = 2)
            
            self.path4_maxpool_3 = nn.MaxPool2D(pool_size =3,padding = 1,strides = 1)
            self.path4_conv_1 = nn.Conv2D(n4_1, kernel_size = 1,activation = 'relu')
            
    def forward(self,x):
        p1 = self.path1_conv_1(x)
        p2 = self.path2_conv_3(self.path2_conv_1(x))
        p3 = self.path3_conv_5(self.path3_conv_1(x))
        p4 = self.path4_conv_1(self.path4_maxpool_3(x))
        return nd.concat(p1,p2,p3,p4,dim =1)
        
```

```{.python .input  n=78}
incp = Inception(n1_1 = 64, n2_1 = 96, n2_3= 128, n3_1 =16, n3_5 = 32, n4_1 = 32)
incp.initialize()
test_data = nd.random.uniform(shape = (32,3,64,64))
print incp(test_data).shape
```

```{.json .output n=78}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(32L, 256L, 64L, 64L)\n"
 }
]
```

```{.python .input  n=153}
class googleNet(nn.Block):
    def __init__(self, num_class, verbose = True, **kwargs):
        self.verbose = verbose
        super(googleNet,self).__init__(**kwargs)
        with self.name_scope():
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(channels = 64,kernel_size = 7, strides = 2, padding = 3, activation = 'relu'),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(channels = 64,kernel_size = 1),
                nn.Conv2D(channels= 192,kernel_size = 3,padding = 1),
                nn.MaxPool2D(pool_size=3,strides = 2)
             )
            
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16, 32, 32),
                Inception(96, 128, 192, 32,96, 64),
                nn.MaxPool2D(pool_size = 3, strides= 2)
            )
            
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )
            
            b6 = nn.Sequential()
            b6.add(
                nn.Flatten()
                ,nn.Dense(num_class)
            )
            
            
            self.net = nn.Sequential()
            self.net.add(b1,b2,b3,b4,b5,b6)
            
    def forward(self,x):
        out = x
        #out = self.b1(x)
        #print('block %d output: %s' % (1, out.shape))
        for i,b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('block %d output: %s' % (i+1, out.shape))
        return out
```

```{.python .input  n=154}
net_test = googleNet(10, verbose=True)
net_test.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
y = net_test(x)
```

```{.json .output n=154}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "block 1 output: (4L, 64L, 23L, 23L)\nblock 2 output: (4L, 192L, 11L, 11L)\nblock 3 output: (4L, 448L, 5L, 5L)\nblock 4 output: (4L, 832L, 2L, 2L)\nblock 5 output: (4L, 1024L, 1L, 1L)\nblock 6 output: (4L, 10L)\n"
 }
]
```

```{.python .input  n=135}
sum([96, 192, 96, 64])
```

```{.json .output n=135}
[
 {
  "data": {
   "text/plain": "448"
  },
  "execution_count": 135,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
