import logging

# 配置日志
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add(a, b):
    result = a + b
    logging.info(f'The sum of {a} and {b} is {result}')
    return result


num1 = 10
num2 = 20
result = add(num1, num2)