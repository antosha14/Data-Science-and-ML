#include <iostream> //This is import in c++
#include <string>

int sum_of_two_numbers(int firstNumber, int secondNumber){
    int twoNumbersSum = firstNumber + secondNumber;
    return twoNumbersSum;
}

int main() 
{/*
    int num_1;
    int num_2;
    std::cout << "Please type 2 numbers, seoarated by spaces" << std::endl;
    std::cin >> num_1 >> num_2; 
    std::cout << sum_of_two_numbers(num_1, num_2) << std::endl;
*/
    //cin thinks that space is a separator for a next variable
    std::string full_name; //we need to import strings in c++
    std::cout << "Please type your name and surname" << std::endl;
    std::getline(std::cin,full_name); //in order to read data with spaces from comand line 1st - reading from, 2nd - putting at
    std::cout << full_name << std::endl;
    std::cout << sizeof(full_name) << std::endl; //size in bytes
    return 0; 
}