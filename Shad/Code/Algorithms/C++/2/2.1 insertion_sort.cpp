// Алгоритм строится на базе текущего сортируемого элемента и индекса предыдшедствующего элемента
// Первый цикл проходится по всем элементам массива, задавая текущий сортируемый элемент и индекс предыдущего элемента
// Для каждого элемента запускается второй цикл, который меняет текущий элемент местами с предшедствующим, если он меньше и отнимает единицу от индекса предыдущего
// После выполнения цикла остаётся только приравнять к key элемент с индексом первого попавшегося в отсортированном массиве меньшего числа плюс один
#include <iostream>
#include <bits/stdc++.h>
using namespace std;
 
// Function to sort an array
// using insertion sort
void insertionSort(int arr[], int n)
{
    int i, key, j;
    for (i = 1; i < n; i++) {
        key = arr[i];
        j = i - 1;
 
            // Move elements of arr[0..i-1],
            // that are greater than key, to one
            // position ahead of their
            // current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}
 
// A utility function to print
// an array of size n
void printArray(int arr[], int n)
{
    int i;
    for (i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
}
 
// Driver code
int main()
{
    int arr[] = { 12, 11, 13, 5, 6 };
    int N = sizeof(arr) / sizeof(arr[0]);
 
    insertionSort(arr, N);
    printArray(arr, N);
 
    return 0;
}