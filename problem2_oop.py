#第二题，面向对象编程，实现一个类叫做QuantumMatrix，含有矩阵乘法方法
class QuantumMatrix:

    def __init__(self, matrix):
        self.matrix= matrix
    #初始化矩阵，matrix是一个二维数组，每个元素是一个元组，元组第一个元素是数值，第二个元素是字符
    #__init__是一个特殊方法，用于初始化对象，第一个参数是self，表示实例本身，后面的参数是传入的参数
    #self.matrix是实例的属性，matrix是传入的参数，两者不同
    #初始化方法只运行一次，当实例化一个对象时，会自动调用这个方法

    def __str__(self):
        return str(self.matrix)
    #__str__是一个特殊方法，用于返回一个对象的描述信息，当使用print函数时，会自动调用这个方法
    #返回的是矩阵的字符串形式，如[[1,2],[3,4]]

    def multiply(self, other):
        #这就是矩阵乘法方法，传入参数是另一个QuantumMatrix对象，从而实现两个矩阵的相乘
        m= len(self.matrix)
        #行a
        n_a= len(self.matrix[0])
        #列a
        n_b= len(other.matrix)
        #行b
        p= len(other.matrix[0])
        #列b
        if n_a != n_b:
            raise ValueError("QuantumDimensionError")
        #判断两个矩阵是否可以相乘，如果矩阵a的列数不等于矩阵b的行数，则无法相乘
        
        c= [[(0, '') for _ in range(p)] for _ in range(m)]
        #初始化一个结果矩阵，大小是m*p，也用到了列表推导式
        for i in range(m):
            for j in range(p):
                totalvalue= 0
                bigchar= ''
                #totalvalue是结果矩阵的元素值，bigchar是结果矩阵的元素字符，均初始化
                for k in range(n_a):
                    num_a, char_a= self.matrix[i][k]
                    #num_a是矩阵a的元素值，char_a是矩阵a的元素字符
                    num_b, char_b= other.matrix[k][j]
                    totalvalue += num_a* num_b
                    bigchar = max(bigchar, char_a, char_b)
                    #数值部分相乘，字符部分取最大值
                c[i][j]= (totalvalue, bigchar)
                #赋值给结果矩阵
            #两层循环分别是遍历结果矩阵的行m和列n
        return QuantumMatrix(c)
        #返回结果矩阵，也是一个QuantumMatrix对象

matrix_a= QuantumMatrix([
    [(2.5,'a'), (-1.8,'a'),(3,'a')],
    [(0.9,'a'), (4.2,'a'),(4,'b')]
])
matrix_b= QuantumMatrix([
    [(1,'m')],
    [(1,'x')],
    [(1,'y')]
])
#实例化两个矩阵对象

print(matrix_a.multiply(matrix_b))
#输出矩阵乘法结果