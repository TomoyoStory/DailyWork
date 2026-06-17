#include <rclcpp/rclcpp.hpp>

#include "visibility_control.h"
#include "image_shm_demo/msg/image_data.hpp"

namespace image_shm_demo
{
    class VecPubNode : public rclcpp::Node
    {
    public:
        COMPOSITION_PUBLIC VecPubNode(const rclcpp::NodeOptions &options)
            : Node("vec_pub_node", options)
        {
            // QoS：只保留最新1帧，可靠传输
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.reliable();
            qos.durability_volatile();

            pub_ = this->create_publisher<image_shm_demo::msg::ImageData>("/image_vec_topic", qos);
            // ✅ 检查零拷贝是否可用
            if (pub_->can_loan_messages())
            {
                RCLCPP_INFO(get_logger(), "✓ Zero-copy (loaned messages) enabled");
            }
            else
            {
                RCLCPP_WARN(get_logger(), "✗ Zero-copy not available, falling back to copy mode");
            }
            timer_ = this->create_wall_timer(std::chrono::milliseconds(33),
                                             std::bind(&VecPubNode::timer_callback, this));

            RCLCPP_INFO(get_logger(), "Publisher ready");
        }

    private:
        void timer_callback()
        {
            // 租借消息内存，零拷贝
            auto loan_msg = pub_->borrow_loaned_message();
            auto &msg = loan_msg.get();

            // 固定尺寸
            const uint32_t W = 1920;
            const uint32_t H = 1080;
            const uint32_t C = 3;
            const size_t total_elem = W * H * C;

            msg.width = W;
            msg.height = H;
            msg.channel = C;

            // 填充渐变测试数据
            static int frame = 0;
            for (size_t i = 0; i < total_elem; i++)
            {
                msg.data[i] = (i + frame) % 255;
            }
            frame++;
            msg.stamp = this->now();

            pub_->publish(std::move(loan_msg));
        }

        rclcpp::Publisher<image_shm_demo::msg::ImageData>::SharedPtr pub_;
        rclcpp::TimerBase::SharedPtr timer_;
    };
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_shm_demo::VecPubNode)