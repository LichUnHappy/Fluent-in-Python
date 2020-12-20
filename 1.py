import copy 

obj = ["will", 28, ["py", "c", "js"]]

# obj2 = obj

# obj2 = copy.copy(obj)

obj2 = copy.deepcopy(obj)


print(f"id of obj {id(obj)}")
print(obj)
print([id(ele) for ele in obj])

print(f"id of obj2 {id(obj2)}")
print(obj2)
print([id(ele) for ele in obj2])

obj[0] = "kill"
obj[2].append("html")
print(f"od of obj {id(obj)}")
print(obj)
print([id(ele) for ele in obj])

print(f"id of obj2 {id(obj2)}")
print(obj2)
print([id(ele) for ele in obj2])