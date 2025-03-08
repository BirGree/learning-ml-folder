class QuantumMatrix:

    def __init__(self, matrix):
        self.matrix= matrix

    def __str__(self):
        return str(self.matrix)

    def multiply(self, other):
        m= len(self.matrix)
        n_a= len(self.matrix[0])
        n_b= len(other.matrix)
        p= len(other.matrix[0])
        if n_a != n_b:
            raise ValueError("QuantumDimensionError")
        #初始化一个结果矩阵，m*p
        c= [[(0, '') for _ in range(p)] for _ in range(m)]
        for i in range(m):
            for j in range(p):
                totalvalue= 0
                bigchar= ''
                for k in range(n_a):
                    num_a, char_a= self.matrix[i][k]
                    num_b, char_b= other.matrix[k][j]
                    totalvalue += num_a* num_b
                    bigchar = max(bigchar, char_a, char_b)
                c[i][j]= (totalvalue, bigchar)
        return QuantumMatrix(c)

matrix_a= QuantumMatrix([
    [(2.5,'a'), (-1.8,'a'),(3,'a')],
    [(0.9,'a'), (4.2,'a'),(4,'b')]
])
matrix_b= QuantumMatrix([
    [(1,'m')],
    [(1,'x')],
    [(1,'y')]
])

print(matrix_a.multiply(matrix_b))