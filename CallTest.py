# __call__的用法可以理解为将一个类直接转换为可以直接进行调用的function
class Person:
    def __call__(self, name):
        print("__call__" + " Hello " + name)
    def hello(self, name):
        print("Hello " + name)

person = Person()
person("Zhang San")
person.hello("Li Si")