#include <chrono>
#include <rclcpp/rclcpp.hpp>

#include "visibility_control.h"
#include "image_shm_demo/msg/image_data.hpp"

namespace image_shm_demo
{
    class VecSubNode : public rclcpp::Node
    {
    public:
        COMPOSITION_PUBLIC VecSubNode(const rclcpp::NodeOptions &options)
            : Node("vec_sub_node", options)
        {
            rclcpp::QoS qos(rclcpp::KeepLast(1));
            qos.reliable();
            sub_ = this->create_subscription<image_shm_demo::msg::ImageData>(
                "/image_vec_topic",
                qos,
                std::bind(&VecSubNode::msg_callback, this, std::placeholders::_1));
            RCLCPP_INFO(get_logger(), "Subscriber waiting vector<int> image data");
            // 检查订阅者是否支持 loaned messages
            if (sub_->can_loan_messages())
            {
                RCLCPP_INFO(get_logger(), "✓ Subscriber loaned messages enabled");
            }
            else
            {
                RCLCPP_WARN(get_logger(), "✗ Subscriber cannot loan messages");
            }

            RCLCPP_INFO(get_logger(), "Subscriber ready, waiting for SHM zero-copy data");
        }

    private:
        void msg_callback(const image_shm_demo::msg::ImageData::SharedPtr msg)
        {
            size_t elem_count = msg->data.size();
            double latency_ms = (this->now() - msg->stamp).nanoseconds() / 1000000.0;
            RCLCPP_INFO(get_logger(),
                        "W:%u H:%u C:%u Elements:%zu Latency:%.3fms",
                        msg->width, msg->height, msg->channel, elem_count, latency_ms);
        }

        rclcpp::Subscription<image_shm_demo::msg::ImageData>::SharedPtr sub_;
    };
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_shm_demo::VecSubNode)