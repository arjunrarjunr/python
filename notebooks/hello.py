import pandas as pd
x = ["Arjun","Aswin","Durga"]
# x = sum(x)
x = pd.DataFrame(x,columns=["Name"])
print(x.Name.str.slice(0,3))


