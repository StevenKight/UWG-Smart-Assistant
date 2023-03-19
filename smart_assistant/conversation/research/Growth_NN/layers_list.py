
def initial_layers():
    layers = [[100],[128, True, 0.5], [64, True, 0.5]]
    return layers

def layer_testing(layers=[], accuracy=-1):
    if layers == [] and accuracy == -1:
        return initial_layers()
    
    # Alter layer structure according to accuracy
    # output everything