#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <ctime>
#include <utility>

using namespace std;

namespace std
{
    template <>
    struct hash<pair<string, string>>
    {
        size_t operator()(const pair<string, string> &p) const noexcept
        {
            size_t h1 = hash<string>{}(p.first);
            size_t h2 = hash<string>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };
}

int main(int argc, char *argv[])
{
    vector<string> states = {"Begin", "Match", "Insertion", "Deletion", "End"};
    vector<double> pi = {1.0, 0.0, 0.0, 0.0, 0.0};
    vector<vector<double>> a(states.size(), vector<double>(states.size(), 0.0));
    vector<unordered_map<string, double>> e(states.size(), unordered_map<string, double>());

    time_t start = time(nullptr);
    ifstream file("inital_values.txt");
    if (!file.is_open())
    {
        cerr << "Error opening file: " << "initial_values.txt" << endl;
        exit(1);
    }

    string value;
    int ind1 = 0, ind2 = 0, puni_a = 1;
    while (file >> value)
    {
        if (value == "TG")
            puni_a = 0;
        if (puni_a == 1)
        {
            double valuelf = stod(value);
            a[ind1][ind2] = valuelf;
            ind2++;
            if (ind2 >= states.size())
            {
                ind2 = 0;
                ind1++;
            }
        }
        else
        {
            string dvotocka;
            double valuelf;
            file >> dvotocka >> valuelf;
            if (value[0] == '-')
            {
                e[2][value] = valuelf;
            }
            else if (value[1] == '-')
            {
                e[3][value] = valuelf;
            }
            else
            {
                e[1][value] = valuelf;
            }
        }
    }

    for (auto &i : a)
    {
        for (auto &j : i)
        {
            cout << j << "\t";
        }
        cout << endl;
    }

    for (int i = 1; i < e.size() - 1; i++)
    {
        cout << "Emissions for state " << states[i] << ":\n";
        for (auto &pair : e[i])
        {
            cout << pair.first << ": " << pair.second << "\n";
        }
        cout << endl;
    }

    time_t end = time(nullptr);
    cout << "Time taken: " << difftime(end, start) << " seconds\n"
         << endl;

    return 0;
}