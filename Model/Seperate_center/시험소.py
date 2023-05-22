from utils.callback_function import best_weight_callback
import logging
from model.network_builder import network_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'test.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

losses = [3,2,1,2,3]
callback = best_weight_callback()

for loss in losses:
    a, b = network_builder('Recurrence_1_SAD')
    a.c = loss
    for param in a.flow_encoder.parameters():
        print(f'Loss = {loss} :: C point :: {a.c}', end=' ')
        print(param[0][0][0])
        break
    callback.add(a, loss)

result_model, result_loss = callback.get_best_model(a)

for param in result_model.flow_encoder.parameters():
    print(f'Loss = {result_loss} :: C point :: {result_model.c}', end=' ')
    print(param[0][0][0])
    break