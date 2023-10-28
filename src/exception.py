import sys
def get_error_detail(error,error_detail:sys):
    _,_,err_tb=error_detail.exc_info()
    filename=err_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script[{0}] in line number [{1}] error message[{2}]".format(filename,err_tb.tb_lineno,str(error))
    return error_message

class Custom_Exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_detail)
        self.error_message=get_error_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message