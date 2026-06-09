import sys

def error_message_detail(error, error_details):

        _,_,exc_tb = error_details.exc_info()

        if exc_tb is not None:
            while exc_tb.tb_next:
                 exc_tb = exc_tb.tb_next

            lineno = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
        
        else:
            lineno = None
            file_name = None

        return (
            f"Error occurred in python script [{file_name}] " 
            f"line number [{lineno}] "
            f"error message [{error}] "
        )
    

class SignalRankException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(str(error_message))

        self.error_message = error_message_detail(
             error_message,
             error_details
        )

    def __str__(self):
         return self.error_message