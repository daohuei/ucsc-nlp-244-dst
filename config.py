from lib2to3.pgen2.token import NAME


BATCH_SIZE = 4
EPOCHS = 10

IS_ADAPTER = True
IS_CURRICULUM = False

NAME = (
    "bart" + "_adapter"
    if IS_ADAPTER
    else "_finetune" + "_cur"
    if IS_CURRICULUM
    else ""
)

