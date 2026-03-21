#ifndef TCP_QLEARNING_H
#define TCP_QLEARNING_H
#include "ns3/rtt-estimator.h"
#include "ns3/nstime.h"
#include "ns3/sequence-number.h"
namespace ns3
{
    class TcpQlearning : public RttEstimator
    {
      public:
        static TypeId GetTypeId(void);
        TcpQlearning();
        Ptr<RttEstimator> Copy() const override;
        TcpQlearning(const TcpQlearning& r);
        void Measurement(Time t) override;
        virtual void SentSeq(SequenceNumber32 seq, uint32_t size);
        void Reset() override;
        void RecordTimeout(Time t);
        Time GetRTO() const override;
      private:
        enum STATE
        {
            SUCCESS,
            FAILURE
        };
        STATE state;
        double q_success;
        double q_failure;
        double alpha_s;
        double alpha_f;
        double gamma_s;
        double gamma_f;
    };
}
#endif
