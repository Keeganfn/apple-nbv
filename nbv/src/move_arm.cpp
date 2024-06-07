#include <rclcpp/rclcpp.hpp>

// Include our custom message type definition.
#include "nbv_interfaces/srv/move_arm.hpp"
#include "nbv_interfaces/srv/move_to_named_target.hpp"
#include "nbv_interfaces/srv/set_sphere_constraint.hpp"

// Moveit
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

// The service callback is going to need two parameters, so we declare that
// we're going to use two placeholders.
using std::placeholders::_1;
using std::placeholders::_2;


// Create a class that inherits from Node.
class MoveArmNode : public rclcpp::Node {
public:
	MoveArmNode();
    // Move group interface
    moveit::planning_interface::MoveGroupInterface move_group_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    void test_arm();

private:
    rclcpp::Service<nbv_interfaces::srv::MoveArm>::SharedPtr arm_service_;
    rclcpp::Service<nbv_interfaces::srv::MoveToNamedTarget>::SharedPtr arm_named_service_;
    rclcpp::Service<nbv_interfaces::srv::SetSphereConstraint>::SharedPtr set_sphere_constraint_service_;
    rclcpp::CallbackGroup::SharedPtr callback_group_subscriber1_; 
    rclcpp::CallbackGroup::SharedPtr callback_group_subscriber2_;
    rclcpp::CallbackGroup::SharedPtr callback_group_subscriber3_;
    void move_to_pose(const std::shared_ptr<nbv_interfaces::srv::MoveArm::Request> request,
		              const std::shared_ptr<nbv_interfaces::srv::MoveArm::Response> response);
    void move_to_named_target(const std::shared_ptr<nbv_interfaces::srv::MoveToNamedTarget::Request> request,
                              const std::shared_ptr<nbv_interfaces::srv::MoveToNamedTarget::Response> response);
    void set_sphere_constraint(
        const std::shared_ptr<nbv_interfaces::srv::SetSphereConstraint::Request> request,
        const std::shared_ptr<nbv_interfaces::srv::SetSphereConstraint::Response> response);
};


MoveArmNode::MoveArmNode() : Node("move_arm_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)), 
                                move_group_(std::shared_ptr<rclcpp::Node>(std::move(this)), "ur_manipulator")
{
    // callback groups
    callback_group_subscriber1_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_subscriber2_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_subscriber3_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Set up services, one for pose goals and one for named target goals
    arm_service_ = this->create_service<nbv_interfaces::srv::MoveArm>(
        "move_arm",
        std::bind(&MoveArmNode::move_to_pose, this, _1, _2), 
        rmw_qos_profile_services_default,
        callback_group_subscriber1_
    );
    arm_named_service_ = this->create_service<nbv_interfaces::srv::MoveToNamedTarget>(
        "move_arm_named_target",
        std::bind(&MoveArmNode::move_to_named_target, this, _1, _2),
        rmw_qos_profile_services_default,
        callback_group_subscriber2_
    );
    set_sphere_constraint_service_ = this->create_service<nbv_interfaces::srv::SetSphereConstraint>(
        "set_sphere_constraint",
        std::bind(&MoveArmNode::set_sphere_constraint, this, _1, _2),
        rmw_qos_profile_services_default,
        callback_group_subscriber3_
    );

    // moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    auto const collision_object = [frame_id =move_group_.getPlanningFrame()] {
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.header.frame_id = frame_id;
        collision_object.id = "box1";
        shape_msgs::msg::SolidPrimitive primitive;

        // Define the size of the box in meters
        primitive.type = primitive.BOX;
        primitive.dimensions.resize(3);
        primitive.dimensions[primitive.BOX_X] = 1;
        primitive.dimensions[primitive.BOX_Y] = 1;
        primitive.dimensions[primitive.BOX_Z] = 0.015;

        // Define the pose of the box (relative to the frame_id)
        geometry_msgs::msg::Pose box_pose;
        box_pose.orientation.w = 1.0;  // We can leave out the x, y, and z components of the quaternion since they are initialized to 0
        box_pose.position.x = 0.0;
        box_pose.position.y = 0.0;
        box_pose.position.z = -0.01;

        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(box_pose);
        collision_object.operation = collision_object.ADD;

        return collision_object;
    }();
    this->planning_scene_interface_.applyCollisionObject(collision_object);
    // vel and acc limits
    this->move_group_.setMaxAccelerationScalingFactor(1.0);
    this->move_group_.setMaxVelocityScalingFactor(1.0);
    RCLCPP_INFO(this->get_logger(), "Move arm server ready");
}

void MoveArmNode::move_to_named_target(const std::shared_ptr<nbv_interfaces::srv::MoveToNamedTarget::Request> request,
                                       const std::shared_ptr<nbv_interfaces::srv::MoveToNamedTarget::Response> response)
{
    // Attempts to move to named target goal 
    if (!this->move_group_.setNamedTarget(request->name)) {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Could not find target " << request->name);
        response->result = false;
    }
    moveit::planning_interface::MoveGroupInterface::Plan goal;
    auto const ok = static_cast<bool>(move_group_.plan(goal));
    
    if (ok){
        this->move_group_.execute(goal);
        response->result = true;
    }
    else{
        RCLCPP_ERROR(this->get_logger(), "Planing failed!");
        response->result = false;
    }

}

void MoveArmNode::move_to_pose(const std::shared_ptr<nbv_interfaces::srv::MoveArm::Request> request,
                               const std::shared_ptr<nbv_interfaces::srv::MoveArm::Response> response)
{
    // Make posestamped message
    // tf2::Quaternion orientation;
    // orientation.setRPY(3.14/2, 3.14, 3.14/2);
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = "base_link";
    // msg.pose.orientation = tf2::toMsg(orientation);;
    msg.pose.position.x = request->goal.position.x;
    msg.pose.position.y = request->goal.position.y;
    msg.pose.position.z = request->goal.position.z;
    msg.pose.orientation.x = request->goal.orientation.x;
    msg.pose.orientation.y = request->goal.orientation.y;
    msg.pose.orientation.z = request->goal.orientation.z;
    msg.pose.orientation.w = request->goal.orientation.w;
    this->move_group_.setPoseTarget(msg, "tool0");
    this->move_group_.setGoalOrientationTolerance(0.1);

    // Attempts to move to pose goal
    moveit::planning_interface::MoveGroupInterface::Plan goal;
    auto const ok = static_cast<bool>(this->move_group_.plan(goal));
    response->result = true;
    if (ok){
        this->move_group_.execute(goal);
    }
    else{
        RCLCPP_ERROR(this->get_logger(), "Planning failed!");
        response->result = false;
    }
}

void MoveArmNode::set_sphere_constraint(
        const std::shared_ptr<nbv_interfaces::srv::SetSphereConstraint::Request> request,
        const std::shared_ptr<nbv_interfaces::srv::SetSphereConstraint::Response> response) 
{
    // RCLCPP_INFO(this-> get_logger(), "%s", request->id);
    auto const collision_object = [request, frame_id=move_group_.getPlanningFrame()] {
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.header.frame_id = "base_link";
        collision_object.id = "sphere" + request->id;//;
        shape_msgs::msg::SolidPrimitive primitive;

        // Define the size of the box in meters
        primitive.type = primitive.SPHERE;
        primitive.dimensions.resize(1);
        primitive.dimensions[0] = request->radius; // sphere only has one dimension in this type

        // Define the pose of the box (relative to the frame_id)
        geometry_msgs::msg::Pose sphere_pose = request->pose;

        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(sphere_pose);

        if (request->remove_from_scene) {
            collision_object.operation = collision_object.REMOVE;
        }
        else {
            collision_object.operation = collision_object.ADD;
        }
        
        return collision_object;
    }();
    bool ok = this->planning_scene_interface_.applyCollisionObject(collision_object);

    if (ok) {
        response->success = true;
        RCLCPP_INFO(this-> get_logger(), "Successfully applied collision object!");
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "Failed to add collision object to the scene.");
        response->success = false;
    }
}

void MoveArmNode::test_arm()
{
    tf2::Quaternion orientation;
    orientation.setRPY(3.14/2, 3.14, 3.14/2);
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = "base_link";
    msg.pose.orientation = tf2::toMsg(orientation);;
    msg.pose.position.x = 0.35;
    msg.pose.position.y = 0.0;
    msg.pose.position.z = 0.6;
    this->move_group_.setPoseTarget(msg, "tool0");
    this->move_group_.setGoalOrientationTolerance(0.1);

    moveit::planning_interface::MoveGroupInterface::Plan goal;
    auto const ok = static_cast<bool>(move_group_.plan(goal));
    if (ok){
        this->move_group_.execute(goal);
    }
    else{
        RCLCPP_ERROR(this->get_logger(), "Planing failed!");
    }
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  auto move_service = std::make_shared<MoveArmNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(move_service);
  executor.spin();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}