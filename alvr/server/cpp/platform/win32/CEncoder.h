#pragma once
#include "shared/d3drender.h"

#include "shared/threadtools.h"

#include <d3d11.h>
#include <wrl.h>
#include <map>
#include <d3d11_1.h>
#include <wincodec.h>
#include <wincodecsdk.h>
#include "alvr_server/Utils.h"
#include "FrameRender.h"
#include "VideoEncoder.h"
#include "VideoEncoderNVENC.h"
#include "VideoEncoderAMF.h"
#ifdef ALVR_GPL
	#include "VideoEncoderSW.h"
#endif
#include "alvr_server/IDRScheduler.h"


	using Microsoft::WRL::ComPtr;

	//----------------------------------------------------------------------------
	// Blocks on reading backbuffer from gpu, so WaitForPresent can return
	// as soon as we know rendering made it this frame.  This step of the pipeline
	// should run about 3ms per frame.
	//----------------------------------------------------------------------------
	class CEncoder : public CThread
	{
	public:
		CEncoder();
		~CEncoder();

		void Initialize(std::shared_ptr<CD3DRender> d3dRender);

		bool CopyToStaging(ID3D11Texture2D *pTexture, uint64_t presentationTime, uint64_t targetTimestampNs);

		virtual void Run();

		virtual void Stop();

		void NewFrameReady(double flVsyncTimeInSeconds);

		void WaitForEncode();

		void OnStreamStart();

		void OnPacketLoss();

		void InsertIDR();

		void CaptureFrame();

	private:
		CThreadEvent m_newFrameReady, m_encodeFinished;
		std::shared_ptr<VideoEncoder> m_videoEncoder;
		bool m_bExiting;
		uint64_t m_presentationTime;
		uint64_t m_targetTimestampNs;
		double m_flVsyncTimeInSeconds;

		std::shared_ptr<FrameRender> m_FrameRender;

		IDRScheduler m_scheduler;
	};

