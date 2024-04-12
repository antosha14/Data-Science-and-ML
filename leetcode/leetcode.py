class Solution:
    def romanToInt(self, s: str) -> int:
        conversion_dict = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        odd_values = {
            "IV": 4,
            "IX": 9,
            "XL": 40,
            "XC": 90,
            "CD": 400,
            "CM": 900,
        }
        target = 0
        cur_index = 0
        next_index = cur_index + 1
        while cur_index < len(s):
            if next_index < len(s) and (s[cur_index] + s[next_index]) in odd_values:
                target += odd_values[(s[cur_index] + s[next_index])]
                cur_index += 2
                next_index = cur_index + 1
            else:
                target += conversion_dict[s[cur_index]]
                cur_index += 1
                next_index = cur_index + 1
        return target
