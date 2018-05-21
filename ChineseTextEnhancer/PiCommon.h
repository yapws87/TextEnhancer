#pragma once
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>

#ifndef PERSONAL_COMPUTER
#include "unistd.h"
#endif

class PiCommon
{
protected:
	std::mutex m_mutex;

public:
	PiCommon();
	~PiCommon();

	std::string getString_fromCmd(std::string cmd);
		
	// Time related functions
	std::string get_current_date();
	std::string get_current_time();
	std::string get_current_time_and_date();
	std::string get_current_day();
	std::string get_yesterday_date();
	void uniSleep(int nSleepTime_milisec);

	bool isDateDiff();
	bool isHourDiff();

	inline std::string getCurrentDate() { return m_currentDate; }
	inline std::string getYesterdayDate() { return m_previousDate; }
	void printStdLog(std::string data);

protected:
	std::string m_currentDate = "";
	std::string m_previousDate = "";
};

