#include "tcp-qlearning.h"
#include "ns3/log.h"
#include "ns3/double.h"

namespace ns3
{
    NS_LOG_COMPONENT_DEFINE ("TcpQlearning");
    NS_OBJECT_ENSURE_REGISTERED (TcpQlearning);
    TypeId
    TcpQlearning::GetTypeId (void)
    {
        static TypeId tid = TypeId ("ns3::TcpQlearning")
            .SetParent<RttEstimator> ()
            .SetGroupName("Internet")
            .AddConstructor<TcpQlearning> ()
        ;
        return tid;
    }
    TcpQlearning::TcpQlearning ()
    {
        NS_LOG_FUNCTION (this);
        state = SUCCESS;
        q_success = 1.0;
        q_failure = 1.0;
        alpha_s = 0.25;
        alpha_f = 0.15;
        gamma_s = -0.5;
        gamma_f = 0.0075;
    }
    TcpQlearning::TcpQlearning (const TcpQlearning& r)
      : RttEstimator (r)
    {
        NS_LOG_FUNCTION (this);
        state = r.state;
        q_success = r.q_success;
        q_failure = r.q_failure;
        alpha_s = r.alpha_s;
        alpha_f = r.alpha_f;
        gamma_s = r.gamma_s;
        gamma_f = r.gamma_f;
    }
    Ptr<RttEstimator>
    TcpQlearning::Copy () const
    {
        return CopyObject<TcpQlearning> (this);
    }
    Time
    TcpQlearning::GetRTO() const
    {
        double rto = (state == SUCCESS) ? q_success : q_failure;
        rto = std::max(0.2, std::min(rto, 60.0));
        return Seconds(rto);
    }  
    void 
    TcpQlearning::Measurement(Time t)
    {
        double delay = t.GetSeconds();
        STATE prevState = state;
        double alpha = (prevState == SUCCESS) ? alpha_s : alpha_f;
        double gamma = (prevState == SUCCESS) ? gamma_s : gamma_f;
        double nextBestQ = std::max(q_success, q_failure);
        double& currentQ = (prevState == SUCCESS) ? q_success : q_failure;
        currentQ = currentQ + alpha * (delay + gamma * nextBestQ - currentQ);
        //currentQ = std::max(0.2, std::min(currentQ, 60.0));
        state = SUCCESS;
        m_estimatedRtt = Seconds(q_success);
        //m_estimatedVariation = Abs(t - m_estimatedRtt) / 2;
        m_nSamples++;
    }
    void 
    TcpQlearning::RecordTimeout(Time t)
    {
        double delay = t.GetSeconds();
        STATE prevState = state;
        double alpha = (prevState == SUCCESS) ? alpha_s : alpha_f;
        double gamma = (prevState == SUCCESS) ? gamma_s : gamma_f;
        double nextBestQ = std::max(q_success, q_failure);
        double& currentQ = (prevState == SUCCESS) ? q_success : q_failure;
        currentQ = currentQ + alpha * (delay + gamma * nextBestQ - currentQ);
        //currentQ = std::max(0.2, std::min(currentQ, 60.0));
        state = FAILURE;
        m_estimatedRtt = Seconds(q_failure);
        //m_estimatedVariation = Abs(Seconds(delay) - m_estimatedRtt) / 2;
    }
    void 
    TcpQlearning::SentSeq(SequenceNumber32 seq, uint32_t size)
    {
        
    }
    void 
    TcpQlearning::Reset()
    {
        state = SUCCESS;
        q_success = 1.0;
        q_failure = 1.0;
        m_estimatedRtt = Time(1.0);
        m_estimatedVariation = Time(0);
        m_nSamples = 0;
    }
}
