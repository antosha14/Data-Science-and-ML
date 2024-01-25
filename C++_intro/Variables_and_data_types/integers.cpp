#include <iostream>

//Different integer notations
//Variable is a named piece of memory, or a link to that piece of memory to be percise
int decimal_st = 15; // integer takes 4 bytes or more, this type of assighnment also chops a number
int decimal_braces {15}; // We can initialise variables like that
int decimal_choped (15.9); // And like that to, but it rounds a number
int octal = 017;
int hexadecimal = 0x0f;
int bynary = 0b00001111;

//But ordinary int also can store negative numbers
signed int decimal_which_can_be_negative = -15; //We use that to indicate that a number can be negative
unsigned int only_positive_variable = -15; // it works, but number is complitly different, DONT USE IT!!
unsigned int only_positive_variable {15};  // use this

//This int modifirers work only with numbers without reminder
signed short int only2bytes_int = {12}; // 2
unsigned long int four_or_eight_bytes_int = {121234123};// 4 or 8

//short and long
    short short_var {-32768} ; //  2 Bytes 
    short int short_int {455} ; // 
    signed short signed_short {122}; //
    signed short int signed_short_int {-456}; // 
    unsigned short int unsigned_short_int {456};
    
    int int_var {55} ; // 4 bytes
    signed signed_var {66};//
    signed int signed_int {77};//
    unsigned int unsigned_int{77};
    
    long long_var {88}; // 4 OR 8 Bytes
    long int long_int {33};
    signed long signed_long {44};
    signed long int signed_long_int {44};
    unsigned long int unsigned_long_int{44};

    long long long_long {888};// 2 long - always 8 Bytes
    long long int long_long_int {999};
    signed long long signed_long_long {444};
    signed long long int signed_long_long_int{1234};
    unsigned long long int unsigned_long_long_int{1234};

int main()
{
    std::cout << decimal_st << std::endl;
    std::cout << decimal_braces << std::endl;
    std::cout << decimal_choped << std::endl;
    std::cout << octal << std::endl;
    std::cout << hexadecimal << std::endl;
    std::cout << bynary << std::endl;
    std::cout << decimal_which_can_be_negative << std::endl;
    std::cout << only_positive_variable << std::endl;
    return 0;
}
 