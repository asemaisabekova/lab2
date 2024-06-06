# main.cs
using System;

public class Matrix
{
    private double[,] values;

    public int Rows { get; }
    public int Columns { get; }

    public Matrix(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        values = new double[rows, columns];
    }

    public Matrix(double[,] data)
    {
        Rows = data.GetLength(0);
        Columns = data.GetLength(1);
        values = data;
    }

    public double this[int i, int j]
    {
        get => values[i, j];
    }

    public Matrix Transpose()
    {
        double[,] result = new double[Columns, Rows];
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result[j, i] = values[i, j];
            }
        }
        return new Matrix(result);
    }

    public static Matrix Zero(int rows, int columns)
    {
        return new Matrix(rows, columns);
    }

    public static Matrix Zero(int n)
    {
        return Zero(n, n);
    }

    public static Matrix Identity(int n)
    {
        Matrix identity = Zero(n);
        for (int i = 0; i < n; i++)
        {
            identity.values[i, i] = 1;
        }
        return identity;
    }

    public override string ToString()
    {
        string result = "";
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                result += $"{values[i, j]} ";
            }
            result += "\n";
        }
        return result;
    }

    public override bool Equals(object obj)
    {
        if (obj == null || !(obj is Matrix))
        {
            return false;
        }

        Matrix other = (Matrix)obj;
        if (Rows != other.Rows || Columns != other.Columns)
        {
            return false;
        }

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                if (values[i, j] != other.values[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public override int GetHashCode()
    {
        // Lazy evaluation of hash code
        int hash = 17;
        unchecked
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    hash = hash * 23 + values[i, j].GetHashCode();
                }
            }
        }
        return hash;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException("Matrices have different dimensions");
        }

        double[,] result = new double[a.Rows, a.Columns];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = a[i, j] + b[i, j];
            }
        }

        return new Matrix(result);
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions.");
        }

        double[,] result = new double[a.Rows, a.Columns];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = a[i, j] - b[i, j];
            }
        }

        return new Matrix(result);
    }

    public static Matrix operator *(Matrix a, double scalar)
    {
        double[,] result = new double[a.Rows, a.Columns];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = a[i, j] * scalar;
            }
        }

        return new Matrix(result);
    }

    public static Matrix operator *(double scalar, Matrix a)
    {
        return a * scalar;
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException("Number of columns in the first matrix must equal the number of rows in the second matrix.");
        }

        double[,] result = new double[a.Rows, b.Columns];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Columns; j++)
            {
                double sum = 0;
                for (int k = 0; k < a.Columns; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        }

        return new Matrix(result);
    }

    public static Matrix operator -(Matrix a)
    {
        return a * -1;
    }

    public static Matrix operator ~(Matrix a)
    {
        return a.Transpose();
    }
}

# operation.cs
using System;
using System.Threading.Tasks;
using System.Threading;

public static class MatrixOperations
{
    public static Matrix Transpose(Matrix matrix)
    {
        return matrix.Transpose();
    }

    public static Matrix MultiplyByScalar(Matrix matrix, double scalar)
    {
        int rows = matrix.Rows;
        int columns = matrix.Columns;
        double[,] result = new double[rows, columns];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = matrix[i, j] * scalar;
            }
        });

        return new Matrix(result);
    }

    public static Matrix Add(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions.");
        }

        int rows = a.Rows;
        int columns = a.Columns;
        double[,] result = new double[rows, columns];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = a[i, j] + b[i, j];
            }
        });

        return new Matrix(result);
    }

    public static Matrix Subtract(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions.");
        }

        int rows = a.Rows;
        int columns = a.Columns;
        double[,] result = new double[rows, columns];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = a[i, j] - b[i, j];
            }
        });

        return new Matrix(result);
    }

    public static Matrix Multiply(Matrix a, Matrix b)
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException("Number of columns in the first matrix must equal the number.");
        }

        int rows = a.Rows;
        int columns = b.Columns;
        int inner = a.Columns;
        double[,] result = new double[rows, columns];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
            {
                double sum = 0;
                for (int k = 0; k < inner; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        });

        return new Matrix(result);
    }

    public static Tuple<Matrix, double> Inverse(Matrix matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        int n = matrix.Rows;
        double determinant = Determinant(matrix);

        if (determinant == 0)
        {
            throw new InvalidOperationException("Matrix is singular.");
        }

        Matrix adjugate = Adjugate(matrix);

        double scalar = 1.0 / determinant;
        Matrix inverse = MultiplyByScalar(adjugate, scalar);

        return Tuple.Create(inverse, determinant);
    }

    private static double Determinant(Matrix matrix)
    {
        if (matrix.Rows == 1)
        {
            return matrix[0, 0];
        }

        if (matrix.Rows == 2)
        {
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
        }

        double det = 0;
        for (int i = 0; i < matrix.Columns; i++)
        {
            det += matrix[0, i] * Cofactor(matrix, 0, i);
        }

        return det;
    }

    private static double Cofactor(Matrix matrix, int row, int col)
    {
        int sign = (row + col) % 2 == 0 ? 1 : -1;
        return sign * Minor(matrix, row, col);
    }

    private static double Minor(Matrix matrix, int row, int col)
    {
        Matrix minorMatrix = new Matrix(matrix.Rows - 1, matrix.Columns - 1);

        for (int i = 0, newRow = 0; i < matrix.Rows; i++)
        {
            if (i == row) continue;
            for (int j = 0, newCol = 0; j < matrix.Columns; j++)
            {
                if (j == col) continue;
                minorMatrix[newRow, newCol] = matrix[i, j];
                newCol++;
            }
            newRow++;
        }

        return Determinant(minorMatrix);
    }

    private static Matrix Adjugate(Matrix matrix)
    {
        int n = matrix.Rows;
        Matrix adjugate = new Matrix(n, n);

        Parallel.For(0, n, i =>
        {
            for (int j = 0; j < n; j++)
            {
                adjugate[j, i] = Cofactor(matrix, i, j);
            }
        });

        return adjugate;
    }
}

# main on site.cs
using System;
using System.IO;
using System.Threading.Tasks;

public class Matrix{
  private readonly double[,] _data;
  public int Rows => _data.GetLength(0);
  public int Columns => _data.GetLength(1);

  public Matrix(double[,] data){
    _data=data;
  }

  public static Matrix GenereteRandomMatrix(int rows,int columns){
    Random rand=new Random();
    double[,] data=new double[rows,columns];
    for(int i=0;i<rows;i++){
      for(int j=0;j<columns;j++){
        data[i,j] = rand.NextDouble()*10;
      }
    }
    return new Matrix(data);
  }

  public static bool CompareArrays(Matrix[] array1,Matrix[] array2){
    if(array1.Length!=array2.Length)
      return false;
    for(int i=0;i<array1.Length;i++){
      if(!array1[i].Equals(array2[i]))
        return false;
    }
    return true;
  }

  public override bool Equals(object obj){
    if(!(obj is Matrix))
      return false;

    Matrix other=(Matrix)obj;
    if(this.Rows!=other.Rows||this.Columns!=other.Columns)
      return false;

    for(int i=0;i<this.Rows;i++){
      for(int j=0;j<this.Columns;j++){
        if(this._data[i,j]!=other._data[i,j])
          return false;
      }
    }
    return true;
  }

  public override int Gethashcode(){
    int hash=17;
    hash=hash*23+this.Rows.GetHashCode();
    hash=hash*23+this.Columns.GetHashCode();
    return hash
  }

  public override string ToString(){
    string result="";
    for(int i=0;i<this.Rows;i++)){
      for(int j=0;j<this.Columns;j++){
        result+=_data[i,j].ToString("F2")+"\t";
      }
      result+=Enviorment.NewLine;
    }
    return result
  }
}

public static class MatrixIO
{
    public static async Task WriteToFileAsync(Matrix matrixA, Matrix matrixB, Stream stream, Func<Stream, Matrix, Task> writeMethod)
    {
        await writeMethod(stream, matrixA);
        await writeMethod(stream, matrixB);
    }

    public static async Task WriteBinaryAsync(Stream stream, Matrix matrix)
    {
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            writer.Write(matrix.Rows);
            writer.Write(matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    writer.Write(matrix[i, j]);
                }
            }
        }
    }

    public static async Task WriteTextAsync(Stream stream, Matrix matrix, string sep = " ")
    {
        using (StreamWriter writer = new StreamWriter(stream))
        {
            await writer.WriteLineAsync($"{matrix.Rows} {matrix.Columns}");
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    await writer.WriteAsync($"{matrix[i, j]}");
                    if (j < matrix.Columns - 1)
                    {
                        await writer.WriteAsync(sep);
                    }
                }
                await writer.WriteLineAsync();
            }
        }
    }

    public static async Task WriteJsonAsync(Stream stream, Matrix matrix)
    {
        double[][] jaggedArray = new double[matrix.Rows][];
        for (int i = 0; i < matrix.Rows; i++)
        {
            jaggedArray[i] = new double[matrix.Columns];
            for (int j = 0; j < matrix.Columns; j++)
            {
                jaggedArray[i][j] = matrix[i, j];
            }
        }

        await JsonSerializer.SerializeAsync(stream, jaggedArray);
    }

    public static async Task<Matrix> ReadFromFileAsync(Stream stream, Func<Stream, Task<Matrix>> readMethod)
    {
        return await readMethod(stream);
    }
}

public static class MatrixOperations
{
    public static async Task MultiplyAndSaveMatricesAsync(Matrix[] matricesA, Matrix[] matricesB, string directory)
    {
        await Task.Run(async () =>
        {
            string binaryDir = Path.Combine(directory, "Бинарный формат");
            string textDir = Path.Combine(directory, "Строковый формат");
            string jsonDir = Path.Combine(directory, "JSON-формат");

            Directory.CreateDirectory(binaryDir);
            Directory.CreateDirectory(textDir);
            Directory.CreateDirectory(jsonDir);

            Task task1 = MultiplyAndSaveMatricesSequentiallyAsync(matricesA, matricesB, binaryDir, "binary_");
            Task task2 = MultiplyAndSaveMatricesSequentiallyAsync(matricesA, matricesB, textDir, "text_");
            Task task3 = ComputeAndSaveScalarProductsAsync(matricesA, matricesB, jsonDir, "json_");

            await Task.WhenAll(task1, task2, task3);

            Console.WriteLine("All matrices saved successfully.");
        });
    }

    private static async Task MultiplyAndSaveMatricesSequentiallyAsync(Matrix[] matricesA, Matrix[] matricesB, string directory, string prefix)
    {
        for (int i = 0; i < matricesA.Length; i++)
        {
            string filePath = Path.Combine(directory, $"{prefix}{i}.bin");
            using (FileStream stream = new FileStream(filePath, FileMode.Create))
            {
                await MatrixIO.WriteToFileAsync(matricesA[i], matricesB[i], stream, MatrixIO.WriteBinaryAsync);
            }
        }
        Console.WriteLine($"Matrices saved in {directory}.");
    }

    private static async Task ComputeAndSaveScalarProductsAsync(Matrix[] matricesA, Matrix[] matricesB, string directory, string prefix)
    {
        for (int i = 0; i < matricesA.Length; i++)
        {
            string filePath = Path.Combine(directory, $"{prefix}{i}.json");
            using (FileStream stream = new FileStream(filePath, FileMode.Create))
            {
                await MatrixIO.WriteToFileAsync(matricesA[i], matricesB[i], stream, MatrixIO.WriteJsonAsync);
            }
        }
        Console.WriteLine($"Matrices saved in {directory}.");
    }

    public static async Task<Matrix[]> ReadMatricesAsync(string directory, string prefix, string extension)
    {
        string[] filePaths = Directory.GetFiles(directory, $"{prefix}*.{extension}");
        Matrix[] matrices = new Matrix[filePaths.Length];
        for (int i = 0; i < filePaths.Length; i++)
        {
            using (FileStream stream = new FileStream(filePaths[i], FileMode.Open))
            {
                matrices[i] = await MatrixIO.ReadFromFileAsync(stream, extension == "bin" ? MatrixIO.ReadBinaryAsync : MatrixIO.ReadJsonAsync);
            }
        }
        return matrices;
    }

    public static async Task<bool

# IO.cs

    using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

public static class MatrixIO
{
    public static async Task WriteTextAsync(Stream stream, Matrix matrix, string sep = " ")
    {
        using (StreamWriter writer = new StreamWriter(stream))
        {
            await writer.WriteLineAsync($"{matrix.Rows} {matrix.Columns}");
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    await writer.WriteAsync($"{matrix[i, j]}");
                    if (j < matrix.Columns - 1)
                    {
                        await writer.WriteAsync(sep);
                    }
                }
                await writer.WriteLineAsync();
            }
        }
    }

    public static async Task<Matrix> ReadTextAsync(Stream stream, string sep = " ")
    {
        using (StreamReader reader = new StreamReader(stream))
        {
            string[] dimensions = (await reader.ReadLineAsync()).Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int rows = int.Parse(dimensions[0]);
            int columns = int.Parse(dimensions[1]);

            double[,] values = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                string[] line = (await reader.ReadLineAsync()).Split(sep, StringSplitOptions.RemoveEmptyEntries);
                for (int j = 0; j < columns; j++)
                {
                    values[i, j] = double.Parse(line[j]);
                }
            }

            return new Matrix(values);
        }
    }

    public static void WriteBinary(Stream stream, Matrix matrix)
    {
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            writer.Write(matrix.Rows);
            writer.Write(matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    writer.Write(matrix[i, j]);
                }
            }
        }
    }

    public static Matrix ReadBinary(Stream stream)
    {
        using (BinaryReader reader = new BinaryReader(stream))
        {
            int rows = reader.ReadInt32();
            int columns = reader.ReadInt32();
            double[,] values = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    values[i, j] = reader.ReadDouble();
                }
            }

            return new Matrix(values);
        }
    }

    public static async Task WriteJsonAsync(Stream stream, Matrix matrix)
    {
        double[][] jaggedArray = new double[matrix.Rows][];
        for (int i = 0; i < matrix.Rows; i++)
        {
            jaggedArray[i] = new double[matrix.Columns];
            for (int j = 0; j < matrix.Columns; j++)
            {
                jaggedArray[i][j] = matrix[i, j];
            }
        }

        await JsonSerializer.SerializeAsync(stream, jaggedArray);
    }

    public static async Task<Matrix> ReadJsonAsync(Stream stream)
    {
        double[][] jaggedArray = await JsonSerializer.DeserializeAsync<double[][]>(stream);

        int rows = jaggedArray.Length;
        int columns = jaggedArray[0].Length;
        double[,] values = new double[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                values[i, j] = jaggedArray[i][j];
            }
        }

        return new Matrix(values);
    }

    public static void WriteToFile(string directory, string fileName, Matrix matrix, Action<Matrix, Stream> writeMethod)
    {
        string filePath = Path.Combine(directory, fileName);
        using (FileStream stream = new FileStream(filePath, FileMode.Create))
        {
            writeMethod(matrix, stream);
        }
    }

    public static async Task WriteToFileAsync(string directory, string fileName, Matrix matrix, Func<Matrix, Stream, Task> writeMethod)
    {
        string filePath = Path.Combine(directory, fileName);
        using (FileStream stream = new FileStream(filePath, FileMode.Create))
        {
            await writeMethod(matrix, stream);
        }
    }

    public static Matrix ReadFromFile(string filePath, Func<Stream, Matrix> readMethod)
    {
        using (FileStream stream = new FileStream(filePath, FileMode.Open))
        {
            return readMethod(stream);
        }
    }

    public static async Task<Matrix> ReadFromFileAsync(string filePath, Func<Stream, Task<Matrix>> readMethod)
    {
        using (FileStream stream = new FileStream(filePath, FileMode.Open))
        {
            return await readMethod(stream);
        }
    }
}

# program.cs

using System;
using System.IO;
using System.Threading.Tasks;

class Program
{
    static void Main(string[] args)
    {
   
        CreateRandomMatrix(3, 3);
        MultiplyMatricesSequentially();
        ComputeScalarProduct();
        WriteMatricesToDirectory();
        ReadMatricesFromDirectory();
        CompareMatricesArrays();
    }

    static void CreateRandomMatrix(int rows, int columns)
    {
        Random rand = new Random();
        double[,] data = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                data[i, j] = rand.NextDouble() * 10;
            }
        }
        Matrix matrix = new Matrix(data);
        Console.WriteLine("Random Matrix:");
        Console.WriteLine(matrix);
    }

    static void MultiplyMatricesSequentially()
    {
        Matrix a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        Matrix b = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix result = MatrixOperations.Multiply(a, b);
        Console.WriteLine("Result of matrix multiplication:");
        Console.WriteLine(result);
    }

    static void ComputeScalarProduct()
    {
        Matrix a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });
        double scalarProduct = MatrixOperations.ComputeScalarProduct(a, b);
        Console.WriteLine($"Scalar product: {scalarProduct}");
    }

    static void WriteMatricesToDirectory()
    {
        Matrix[] matrices = new Matrix[10];
        for (int i = 0; i < 10; i++)
        {
            matrices[i] = Matrix.GenerateRandomMatrix(3, 3);
        }

        MatrixIO.WriteToDirectory(matrices, ".", "matrix", "txt", MatrixIO.WriteText);
        Console.WriteLine("Matrices written to directory.");
    }

    static void ReadMatricesFromDirectory()
    {
        Matrix[] matrices = MatrixIO.ReadFromDirectory(".", "matrix", "txt", MatrixIO.ReadText);
        Console.WriteLine("Matrices read from directory:");
        foreach (var matrix in matrices)
        {
            Console.WriteLine(matrix);
        }
    }

    static void CompareMatricesArrays()
    {
        Matrix[] array1 = new Matrix[2];
        array1[0] = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        array1[1] = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });

        Matrix[] array2 = new Matrix[2];
        array2[0] = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        array2[1] = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });

        bool equal = Matrix.CompareArrays(array1, array2);
        Console.WriteLine($"Matrices arrays are {(equal ? "equal" : "not equal")}.");
    }
}
