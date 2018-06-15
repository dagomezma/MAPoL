#ifndef STRINGS_H_
#define STRINGS_H_

#include <string>
#include <sstream>

using namespace std;
/*
inline __host__ string trim(const string& str, const string& whitespace = " ") {
    return remove_if(str.begin(), str.end(), whitespace);
}

inline __host__ string rightMember(const string& str, const string& op = "=") {
    return str.substr(str.find(op) + 1, str.length());
}
*/
template<typename type>
inline __host__ string to_string(type a) {
	string s;
	stringstream ss;
	ss << a;
	ss >> s;
	return s;
}

#endif /* STRINGS_H_ */
