##### MNIST의 정보를 불러온다.
```python
from Dataset.mnist import *
```

##### load_mnist 함수를 통해 MNIST의 자료를 불러온다.
```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)
```

##### 여기서 one_hot_lable을 True, False 했을 때의 target값이 다르게 나온다.
##### True를 넣게 되면 데이터가 60000개 있을 경우 각 데이터가 나타내는 값(0~9)이 [60000, ]의 형태로 나오게 된다.

##### False를 넣게 되면 데이터가 60000개 있을 경우 각 데이터가 나타내는 값은 [10,] 의 형태로 나오게 된다. 즉 60000개의 데이터가 있을 경우 [60000, 10]의 형태로 나온다.

##### TensorFlow를 쓰는 이유? <출처 : https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/pros/>
##### 파이썬에서 효율적인 수치 연산을 하기 위해, 우리는 다른 언어로 구현된 보다 효율이 높은 코드를 사용하여 행렬곱 같은 무거운 연산을 수행하는 NumPy등의 라이브러리를 자주 사용합니다. 그러나 아쉽게도, 매 연산마다 파이썬으로 다시 돌아오는 과정에서 여전히 많은 오버헤드가 발생할 수 있습니다. 이러한 오버헤드는 GPU에서 연산을 하거나 분산 처리 환경같은, 데이터 전송에 큰 비용이 발생할 수 있는 상황에서 특히 문제가 될 수 있습니다.
##### 텐서플로우 역시 파이썬 외부에서 무거운 작업들을 수행하지만, 텐서플로우는 이런 오버헤드를 피하기 위해 한 단계 더 나아간 방식을 활용합니다. 파이썬에서 하나의 무거운 작업을 독립적으로 실행하는 대신, 텐서플로우는 서로 상호작용하는 연산간의 그래프를 유저가 기술하도록 하고, 그 연산 모두가 파이썬 밖에서 동작합니다 (이러한 접근 방법은 다른 몇몇 머신러닝 라이브러리에서 볼 수 있습니다).
##### TensorFlow는 계산을 위해 고효율의 C++ 백엔드(backend)를 사용합니다. 이 백엔드와의 연결을 위해 TensorFlow는 세션(session)을 사용합니다. 일반적으로 TensorFlow 프로그램은 먼저 그래프를 구성하고, 그 이후 그래프를 세션을 통해 실행하는 방식을 따릅니다. 
##### InteractiveSession 클래스를 사용하는 이유 - TensorFlow 코드를 보다 유연하게 작성할 수 있게 해 주는 InteractiveSession 클래스를 사용할 것입니다. 이 클래스는 계산 그래프(computation graph)를 구성하는 작업과 그 그래프를 실행하는 작업을 분리시켜 줍니다. 즉, InteractiveSession을 쓰지 않는다면, 세션을 시작하여 그래프를 실행하기 전에 이미 전체 계산 그래프가 구성되어 있어야 하는 것입니다.


##### TensorFlow를 써서 간단하게 단일층 layer는 구현하였으나 MultiLayer는 어떻게 구현을 해야하는가?
```python
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=True)
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        x = tf.placeholder("float", [None, 784]) #placeholder의 정확한 역할은 무엇인가?
        
        W1 = tf.Variable(tf.zeros([784,50]))
        b1 = tf.Variable(tf.zeros([50]))
        
        
        W2 = tf.Variable(tf.zeros([50, 10]))
        b2 = tf.Variable(tf.zeros([10]))
        
        #y1 = tf.nn.softmax(tf.matmul(x, W1)+b1)
        y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
        
        y2 = tf.nn.softmax(tf.matmul(y1,W2)+b2) 
        
        t = tf.placeholder(tf.float32, [None, 10])
         
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y2), reduction_indices=[1]))
         
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
             
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
         
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
             
        correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(t,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         
        print(sess.run(accuracy, feed_dict={x:mnist.test.images, t:mnist.test.labels})) #placeholder에 저장한 값을 run할 때 넣어준다.
```
##### 위와 같이 구현하였더니 정답률이 0.1135가 나온다. 그 이유는 무엇인가? weight이 다음과 같이 정의 되었다
```python
 W1 = tf.Variable(tf.zeros([784,50]))
 ```
##### 왜 weight을 초기값으로 0을 주면 학습이 잘 안되는지 고찰해볼 필요가 있다.

##### 다음과 같은 조건을 주었더니 정확도가 0.9764까지 나왔다.
```python

        epoch = 10000

        W1 = tf.Variable(tf.random_uniform([784,50], -0.01, 0.01)) #처음에 weight를 tf.zeros[784,50] 하는 것보다 값이 훨씬 잘 나옴, 초기값의 중요성
        b1 = tf.Variable(tf.zeros([50]))
        
        
        W2 = tf.Variable(tf.random_uniform([50,10], -0.01, 0.01))
        b2 = tf.Variable(tf.zeros([10]))
        y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
        y2 = tf.nn.softmax(tf.matmul(y1,W2)+b2)

        t = tf.placeholder(tf.float32, [None, 10])
```

##### 은닉층을 2개로 했더니 또다시 0.098이라는 황당한 값이 나온다. epoch를 10으로 줘도 0.1135가 나온다.
##### weight의 초기값을 아래와 같이 주고 돌렸다.
```python
        W1 = tf.Variable(tf.random_uniform([784,50], -0.08, 0.08)) #처음에 weight를 tf.zeros[784,50] 하는 것보다 값이 훨씬 잘 나옴, 초기값의 중요성
        b1 = tf.Variable(tf.zeros([50]))
        
        
        W2 = tf.Variable(tf.random_uniform([50,50], -0.08, 0.08))
        b2 = tf.Variable(tf.zeros([50]))
        
        W3 = tf.Variable(tf.random_uniform([50,10], -0.08, 0.08))
```
##### epoch를 1000으로 했을 때 0.9587이 나온다. 10000으로 하면 역시 0.098이 나온다. 왜그럴까??? 고민을 해봐야 한다.