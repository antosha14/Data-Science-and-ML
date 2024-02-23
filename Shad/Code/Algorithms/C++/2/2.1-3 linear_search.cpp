#include <bits/stdc++.h>
using namespace std;
#include <iostream>

void linearSearch(int arr[], int N, int searched_num)
{
    int i, key, j, searched_index;
    for (i = 0; i < N; i++) {
        if(arr[i]==searched_num){
            searched_index = i;
        }
    }
    cout << arr[searched_index] << endl;
}
 
// A utility function to print
// an array of size n
void printArray(int arr[], int N)
{
    int i;
    for (i = 0; i < N; i++)
        cout << arr[i] << " ";
    cout << endl;
}
 
// Driver code
int main()
{
    int arr[] = { 12, 11, 13, 5, 6, 12, 3, 4, 55, 12, 345, 123};
    int N = sizeof(arr) / sizeof(arr[0]);
    int searched_num = 6;
    linearSearch(arr, N, searched_num);
    printArray(arr, N);

    return 0;
}