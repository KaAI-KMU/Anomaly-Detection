import logging

class best_weight_callback:
    def __init__(self, name, debug = False):
        self.weight = None
        self.c = None
        self.loss = 1e9
        self.task = name
        self.debug = debug
    def add(self, model, loss_value):
        # 이전에 best_weight가 없는 경우
        if self.weight == None:
            logging.info(f'CallBack_Function\tBest weight\tFisrt {self.task} Model')
            self.loss = loss_value
            self.weight = model.state_dict()
            self.c = model.c
        # 이전의 loss값이 이번의 loss값 보다 큰 경우
        elif self.loss > loss_value:
            logging.info(f'CallBack_Function\tBest weight\tBest {self.task} Model Detected\t::\tLoss :: {self.loss} -> {loss_value}')
            self.loss = loss_value
            self.weight = model.state_dict()
            self.c = model.c
    def get_best_model(self, model):
        model.load_state_dict(self.weight)
        model.c = self.c
        logging.info(f'CallBack_Function\tBest weight\tReturn Best {self.task} Model :: loss = {self.loss}')
        if self.debug:
            return model, self.loss
        else:
            return model