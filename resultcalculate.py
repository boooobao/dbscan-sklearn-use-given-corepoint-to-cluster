
def cal_accuracy(corr,resu):
    amount = len(corr)
    count = 0
    for x,y in corr,resu:
        if x!=y:
            count = count + 1
    return (amount - count)/amount


def precision_score(y_true, y_pred,i):
        return ((y_true == i) * (y_pred == i)).sum() / (y_pred == i).sum()

def recall_score(y_true, y_pred,i):
        return ((y_true == i) * (y_pred == i)).sum() / (y_true == i).sum()

def f1_score(y_true, y_pred):
        num = 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred)
        deno = (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
        return num / deno