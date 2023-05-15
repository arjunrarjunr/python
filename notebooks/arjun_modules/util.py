class stats:
    def __init__(self):
        pass
    def frequency_distribution(self,input_data_list:list|set|tuple|str):
        frequency_dict = {}
        if type(input_data_list) is list or type(input_data_list) is set or type(input_data_list) is tuple or type(input_data_list) is str:
            for i in input_data_list:
                try:
                    frequency_dict[i] += 1
                except KeyError:
                    frequency_dict[i] = 1
            return frequency_dict
        else:
            raise TypeError("The input data should be either list or tuple or set or string.")

    
    def print_dictionary(self,input:dict):
        if type(input) is dict:
            for key,value in input.items():
                print("key :",key,"value :",value)
        else:
            raise TypeError("The input should be of type dictionary.")

    def mean(self,input_data_list:list):
        
        if type(input_data_list) is list:
            try:
                mean = 0
                for i in input_data_list:
                    mean += i
                mean /= len(input_data_list)
                return mean
            except TypeError:
                raise TypeError("The input data list should contain only numbers.")
        else:
            raise TypeError("The input data should be either list.")

    def median(self,input_data_list:list):

        if type(input_data_list) is list:
            from math import ceil,floor
            try:
                median = 0
                if len(input_data_list) % 2 == 1:
                    input_data_list = sorted(input_data_list)
                    median = input_data_list[ceil(len(input_data_list)/2)]
                else:
                    input_data_list = sorted(input_data_list)
                    median = self.mean([input_data_list[int(len(input_data_list)/2)],input_data_list[int(len(input_data_list)/2-1)]])
                return median                    
            except TypeError:
                raise TypeError("The input data list should contain only numbers.")
        else:
            raise TypeError("The input data should be either list.")
