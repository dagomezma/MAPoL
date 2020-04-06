/*
 * writelog.h
 *
 *  Created on: 04/05/2017
 *      Author: vincent
 */

#ifndef WRITELOG_H_
#define WRITELOG_H_

#ifndef SQL_LOG_FILE
void writeLog(string text){
	time_t now = time(0);
	string temp = asctime(localtime(&now));
	cout << "[" << temp.substr(0,temp.size()-1) << "] - " << text <<  endl;
	logFile << "[" << temp.substr(0,temp.size()-1) << "] - " << text <<  endl;
	logFile.flush();
}

void writeLogWithNoDate(string text){
	cout  << text <<  endl;
	logFile << text <<  endl;
	logFile.flush();
}
#else
#define writeLogWithNoDate writeLog
void writeLog(string text, bool comment = true){
	if (comment) {
		cout << text <<  endl;
		logFile << "# ";
	}
	logFile << text <<  endl;
	logFile.flush();
}
#endif

#endif /* WRITELOG_H_ */
