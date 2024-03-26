class MyNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
			
			
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "image/mynode2"

    def test(self):
        return ()


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstNode": "My First Node"
}
