import sys

def error_msg_Details(error, error_details:sys):
    """Format error message with details"""
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"
    return error_message

class CustomError(Exception):
    """Base class for custom exceptions"""
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_msg_Details(error_message, error_details)
        
    
    def __str__(self):
        return self.error_message
    

# if __name__ == "__main__":
        
#     try:
#         10/0
#     except Exception as e:
#         raise CustomError(e, sys)