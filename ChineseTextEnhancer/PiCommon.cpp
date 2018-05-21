#include "PiCommon.h"



PiCommon::PiCommon()
{
}


PiCommon::~PiCommon()
{
}

std::string PiCommon::getString_fromCmd(std::string cmd)
{
	std::string data;
	FILE * stream;
	const int max_buffer = 256;
	char buffer[max_buffer];
	cmd.append(" 2>&1");

#ifdef _WIN32
	stream = _popen(cmd.c_str(), "r");
	if (stream) {
		while (!feof(stream))
			if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
		_pclose(stream);
	}
#else
	stream = popen(cmd.c_str(), "r");
	if (stream) {
		while (!feof(stream))
			if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
		pclose(stream);
	}

#endif
	return data;
}

std::string PiCommon::get_current_date()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	
	std::stringstream ss;
	m_mutex.lock();
	struct tm *printedTime = new tm;
#ifdef PERSONAL_COMPUTER
	localtime_s(printedTime, &in_time_t);
#else
	localtime_r(&in_time_t, printedTime);
#endif
	ss << std::put_time(printedTime, "%Y-%m-%d");
	m_mutex.unlock();
	delete printedTime;

	std::string return_str = ss.str();
	return return_str;
}

std::string PiCommon::get_current_time()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	m_mutex.lock();
	struct tm *printedTime = new tm;
#ifdef PERSONAL_COMPUTER
	localtime_s(printedTime, &in_time_t);
#else
	localtime_r(&in_time_t, printedTime);
#endif

	ss << std::put_time(printedTime, "%X");
	m_mutex.unlock();
	delete printedTime;

	std::string return_str = ss.str();
	return return_str;
}


std::string PiCommon::get_current_time_and_date()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	m_mutex.lock();
	struct tm *printedTime = new tm;

#ifdef PERSONAL_COMPUTER
	localtime_s(printedTime, &in_time_t);
#else
	localtime_r(&in_time_t, printedTime);
#endif
	ss << std::put_time(printedTime, "%Y-%m-%d %X");
	m_mutex.unlock();
	delete printedTime;

	std::string return_str = ss.str();
	return return_str;
}

std::string PiCommon::get_yesterday_date()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	in_time_t = in_time_t - (24 * 60 * 60);

	std::stringstream ss;
	m_mutex.lock();
	struct tm *printedTime = new tm;

#ifdef PERSONAL_COMPUTER
	localtime_s(printedTime, &in_time_t);
#else
	localtime_r( &in_time_t, printedTime);
#endif
	ss << std::put_time(printedTime, "%Y-%m-%d");
	m_mutex.unlock();
	delete printedTime;

	std::string return_str = ss.str();
	return return_str;
}



std::string PiCommon::get_current_day()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	m_mutex.lock();
	struct tm *printedTime = new tm;

#ifdef PERSONAL_COMPUTER
	localtime_s(printedTime, &in_time_t);
#else
	localtime_r(&in_time_t, printedTime);
#endif
	ss << std::put_time(printedTime, "%d");
	m_mutex.unlock();
	delete printedTime;

	std::string return_str = ss.str();
	return return_str;
}

bool PiCommon::isDateDiff()
{
	if (m_currentDate != "")
	{
		std::string cur_date = get_current_date();
		if (m_currentDate != cur_date)
		{
			std::cout << m_currentDate << "  " << cur_date << "\n";
			m_previousDate = m_currentDate;
			m_currentDate = cur_date;
			return true;
		}

	}
	else
	{
		m_currentDate = get_current_date();
	}
	return false;
}


bool PiCommon::isHourDiff()
{
	return false;
}

void PiCommon::printStdLog(std::string data)
{
	std::cout << "[" << get_current_time_and_date() << "]\t";
	std::cout << data << std::endl;
}


void PiCommon::uniSleep(int nSleepTime_milisec)
{
#ifndef PERSONAL_COMPUTER
	usleep(nSleepTime_milisec);
#else
	std::this_thread::sleep_for(std::chrono::milliseconds(nSleepTime_milisec));
#endif
}