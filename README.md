# makeyourownneuralnetwork
Code for the [Make Your Own Neural Network book](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/r)

blog: https://makeyourownneuralnetwork.blogspot.com/

1. Run uwsgi 
uwsgi --http-socket :9091 --plugin python3 --wsgi-file ./test_service.py

2. Run sketch for processing test_proccessing.pda

3. Input digit in the processing window. Send to neural net by pressing enter. 

![test](https://github.com/tech0x/makeyourownneuralnetwork/blob/master/image.jpg?raw=true)
