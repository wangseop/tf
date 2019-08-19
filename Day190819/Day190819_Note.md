## Tensorflow Session
* Session은 Tensor Model 안에 들어가는  Graph를 해석 할 수 있다.
* 즉 Compile에 대한 Graph, fit에 대한 Graph, evaluate/ predict에 대한 graph를 Session에서 run 하게 되면
* 그 그래프에 해당하는 내용이 출력된다. 
* Keras에서의 Compile, fit, evaluate, predic는 모두  run()을 통해 해석된다.