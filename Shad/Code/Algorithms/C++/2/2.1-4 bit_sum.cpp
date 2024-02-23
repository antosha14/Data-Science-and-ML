// Побитовую сумму делаем через остаток от деления на 2
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int first_number_array [] {1, 0 , 1, 1, 1, 0, 0, 1};
int second_number_array [] {1, 0 , 0, 1, 0, 0, 0, 1};
int sum_array [] {0, 0, 0, 0, 0, 0, 0, 0, 0};

int s = sizeof(first_number_array) / sizeof(first_number_array[0]);
int reminder {0};

int main()
{   
   int n {s};
   for (n; n>=0; n--){
        sum_array[n] = (first_number_array[n-1] + second_number_array [n-1] + reminder)%2;
        if ((first_number_array[n-1] + second_number_array [n-1] + reminder)>=2){
            reminder = 1;
        } else {
            reminder = 0;
        }
    }
    sum_array[0] = reminder;

    //Print first arr
    int i;
    for (i=0; i<s; i++){
        cout << first_number_array[i];
    }
    cout << endl; 

    //Print second arr
    for (i=0; i<s; i++){
        cout << second_number_array[i];
    }
    cout << endl; 

    //Print sum arr
    for (i=0; i<=s; i++){
        cout << sum_array[i];
    }


    return 0;
}