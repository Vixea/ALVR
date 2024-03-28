//===================== Copyright (c) Valve Corporation. All Rights Reserved. ======================
//
// Helper classes for working with threads.
//
//==================================================================================================
#pragma once

#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

#define THREAD_PRIORITY_MOST_URGENT 15

class CThread
{
public:
	CThread();
	virtual ~CThread();
	virtual bool Init() { return true; }
	virtual void Run() = 0;
	void Start();
	void Join();
private:
	std::thread *m_pThread;
};

// Changes by Vixea
#ifdef __linux
class CThreadEvent
{
	public:
		CThreadEvent( bool bManualReset = false );
		~CThreadEvent();
		bool Wait( uint32_t nTimeoutMs = std::numeric_limits<uint32_t>::max() );
		bool Set();
		bool Reset();
	private:
};
#endif
// End of changes

#ifdef _WIN32
class CThreadEvent
{
public:
	CThreadEvent( bool bManualReset = false );
	~CThreadEvent();
	bool Wait( uint32_t nTimeoutMs = INFINITE );
	bool Set();
	bool Reset();
private:
	HANDLE m_hSyncObject;
};
#endif
