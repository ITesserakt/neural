//! ORIGINAL CODE BELONGS TO: https://github.com/djrakita/ad_trait

use crate::differentiation::{Derivatives, Indexed};
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use smallvec::{smallvec, SmallVec};
use std::cell::OnceCell;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::ptr::NonNull;
use std::sync::RwLock;

/// A type for Reverse-mode Automatic Differentiation.
///
/// `Adr` stores its current value and a reference to its position (node index)
/// in a global computation graph. This allows for computing gradients by rebuilding
/// the chain of operations and backpropagating adjoints.
pub struct Adr<T> {
    /// The primary value.
    pub number: T,
    /// The index of the node representing this value in the computation graph.
    pub(super) node_idx: NodeIdx,
}

impl<T: Debug> Debug for Adr<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("adr")
            .field("value", &self.number)
            .field("node_idx", &self.node_idx)
            .finish()
    }
}

impl<T> Adr<T> {
    /// Creates a new variable in the global computation graph.
    ///
    /// # Arguments
    /// * `value` - The initial value.
    /// * `reset_computation_graph` - If true, the global graph will be cleared before adding this variable.
    #[inline]
    pub fn variable(value: T) -> Self
    where
        T: Zero + Clone + 'static,
    {
        GlobalComputationGraph::get().spawn_value(value)
    }

    #[inline]
    pub const fn constant(value: T) -> Self {
        Self {
            number: value,
            node_idx: NodeIdx::Constant,
        }
    }

    #[inline]
    pub fn is_constant(&self) -> bool {
        match self.node_idx {
            NodeIdx::Constant => true,
            _ => false,
        }
    }
}

impl<T: Clone> Clone for Adr<T> {
    fn clone(&self) -> Self {
        Self {
            number: self.number.clone(),
            node_idx: self.node_idx,
        }
    }
}
impl<T: Copy> Copy for Adr<T> {}

impl<T: ScalarOperand> ScalarOperand for Adr<T> {}

pub struct ComputationGraph<T> {
    add_idx: RwLock<usize>,
    nodes: RwLock<Vec<ComputationGraphNode<T>>>,
    derivatives_pool: object_pool::Pool<Vec<T>>,
}

impl<T> ComputationGraph<T> {
    fn new() -> Self {
        Self {
            add_idx: RwLock::new(0),
            nodes: RwLock::new(vec![]),
            derivatives_pool: object_pool::Pool::new(1, || Vec::new()),
        }
    }

    fn reset(&self) {
        *self.add_idx.write().expect("error") = 0;
        self.nodes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    pub fn get_backwards_mode_grad(&self, node_idx_enum: NodeIdx) -> Derivatives<'_, T>
    where
        T: Float + AddAssign,
    {
        if let NodeIdx::Constant = node_idx_enum {
            unreachable!("Adr has no parent operations to build derivative from")
        }

        let nodes = self.nodes.read().unwrap();
        let add_idx = *self.add_idx.read().unwrap();
        let mut adjoints = self.derivatives_pool.pull(|| Vec::new());
        adjoints.clear();
        adjoints.resize(add_idx, T::zero());

        match node_idx_enum {
            NodeIdx::Constant => unreachable!(),
            NodeIdx::Idx(node_idx) => adjoints[node_idx] = T::one(),
        };

        for node_idx in (0..add_idx).rev() {
            let node = &nodes[node_idx];
            let parent_adjoints = node
                .node_type
                .get_derivatives_wrt_parents(node.parent_0, node.parent_1);
            if parent_adjoints.len() == 1 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
            } else if parent_adjoints.len() == 2 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                let parent_1_idx = node.parent_1_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
                if parent_1_idx != NodeIdx::Constant {
                    adjoints[parent_1_idx.get_idx()] += curr_adjoint * parent_adjoints[1];
                }
            }
        }

        Derivatives { adjoints }
    }

    #[inline(always)]
    fn spawn_variable(&self, value: T) -> Adr<T>
    where
        T: Clone + Zero,
    {
        let mut nodes = self.nodes.write().expect("error");
        let mut add_idx = self.add_idx.write().expect("error");
        let node_idx = *add_idx;
        let l = nodes.len();

        let node = ComputationGraphNode {
            node_idx,
            node_type: NodeType::Constant,
            value: value.clone(),
            parent_0: None,
            parent_1: None,
            parent_0_idx: None,
            parent_1_idx: None,
        };

        if node_idx >= l {
            nodes.push(node);
        } else {
            nodes[node_idx] = node;
        }

        let out = Adr {
            number: value,
            node_idx: NodeIdx::Idx(node_idx),
        };

        *add_idx += 1;

        out
    }

    #[inline(always)]
    fn add_node(
        &self,
        node_type: NodeType,
        value: T,
        parent_0: Option<T>,
        parent_1: Option<T>,
        parent_0_idx: Option<NodeIdx>,
        parent_1_idx: Option<NodeIdx>,
    ) -> Adr<T>
    where
        T: Clone + Zero,
    {
        match (parent_0_idx, parent_1_idx) {
            (Some(NodeIdx::Constant), Some(NodeIdx::Constant)) => {
                return Adr {
                    number: value,
                    node_idx: NodeIdx::Constant,
                };
            }
            (Some(NodeIdx::Constant), None) => {
                return Adr {
                    number: value,
                    node_idx: NodeIdx::Constant,
                };
            }
            _ => {}
        };

        let mut nodes = self.nodes.write().expect("error");
        let mut add_idx = self.add_idx.write().expect("error");
        let node_idx = *add_idx;
        let l = nodes.len();
        if node_idx >= l {
            nodes.push(ComputationGraphNode {
                node_idx,
                node_type,
                value: value.clone(),
                parent_0,
                parent_1,
                parent_0_idx,
                parent_1_idx,
            });
        } else {
            nodes[*add_idx] = ComputationGraphNode {
                node_idx,
                node_type,
                value: value.clone(),
                parent_0,
                parent_1,
                parent_0_idx,
                parent_1_idx,
            }
        }

        let out = Adr {
            number: value,
            node_idx: NodeIdx::Idx(node_idx),
        };

        *add_idx += 1;

        out
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ComputationGraphNode<T> {
    node_idx: usize,
    node_type: NodeType,
    value: T,
    parent_0: Option<T>,
    parent_1: Option<T>,
    parent_0_idx: Option<NodeIdx>,
    parent_1_idx: Option<NodeIdx>,
}

#[derive(Clone, Debug, Copy, Default)]
pub enum NodeType {
    #[default]
    Constant,
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Abs,
    Signum,
    Max,
    Min,
    Atan2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Fract,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Log,
    Ln,
    Sqrt,
    Exp,
    Powf,
}

impl<T: Zero> Default for ComputationGraphNode<T> {
    fn default() -> Self {
        Self {
            node_idx: Default::default(),
            node_type: Default::default(),
            value: T::zero(),
            parent_0: Default::default(),
            parent_1: Default::default(),
            parent_0_idx: Default::default(),
            parent_1_idx: Default::default(),
        }
    }
}

impl NodeType {
    fn get_derivatives_wrt_parents<T>(
        &self,
        parent_0: Option<T>,
        parent_1: Option<T>,
    ) -> SmallVec<[T; 2]>
    where
        T: Float,
    {
        match self {
            NodeType::Constant => {
                smallvec!()
            }
            NodeType::Add => {
                smallvec!(T::one(), T::one())
            }
            NodeType::Mul => {
                smallvec!(parent_1.unwrap(), parent_0.unwrap())
            }
            NodeType::Sub => {
                smallvec!(T::one(), -T::one())
            }
            NodeType::Div => {
                smallvec!(
                    T::one() / parent_1.unwrap(),
                    -parent_0.unwrap() / (parent_1.unwrap() * parent_1.unwrap())
                )
            }
            NodeType::Neg => {
                smallvec!(-T::one())
            }
            NodeType::Abs => {
                let val = parent_0.unwrap();
                if val >= T::zero() {
                    smallvec!(T::one())
                } else {
                    smallvec!(-T::one())
                }
            }
            NodeType::Signum => {
                smallvec!(T::zero())
            }
            NodeType::Max => {
                if parent_0.unwrap() >= parent_1.unwrap() {
                    smallvec!(T::one(), T::zero())
                } else {
                    smallvec!(T::zero(), T::one())
                }
            }
            NodeType::Min => {
                if parent_0.unwrap() <= parent_1.unwrap() {
                    smallvec!(T::one(), T::zero())
                } else {
                    smallvec!(T::zero(), T::one())
                }
            }
            NodeType::Atan2 => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                smallvec!(
                    rhs / (lhs * lhs + rhs * rhs),
                    -lhs / (lhs * lhs + rhs * rhs)
                )
            }
            NodeType::Floor => {
                smallvec!(T::zero())
            }
            NodeType::Ceil => {
                smallvec!(T::zero())
            }
            NodeType::Round => {
                smallvec!(T::zero())
            }
            NodeType::Trunc => {
                smallvec!(T::zero())
            }
            NodeType::Fract => {
                smallvec!(T::one())
            }
            NodeType::Sin => {
                smallvec!(T::cos(parent_0.unwrap()))
            }
            NodeType::Cos => {
                smallvec!(T::sin(-parent_0.unwrap()))
            }
            NodeType::Tan => {
                let c = T::cos(parent_0.unwrap());
                smallvec!(T::one() / (c * c))
            }
            NodeType::Asin => {
                smallvec!(T::one() / T::sqrt(T::one() - parent_0.unwrap() * parent_0.unwrap()))
            }
            NodeType::Acos => {
                smallvec!(-T::one() / T::sqrt(T::one() - parent_0.unwrap() * parent_0.unwrap()))
            }
            NodeType::Atan => {
                smallvec!(T::one() / (parent_0.unwrap() * parent_0.unwrap() + T::one()))
            }
            NodeType::Sinh => {
                smallvec!(T::cosh(parent_0.unwrap()))
            }
            NodeType::Cosh => {
                smallvec!(T::sinh(parent_0.unwrap()))
            }
            NodeType::Tanh => {
                let c = T::cosh(parent_0.unwrap());
                smallvec!(T::one() / (c * c))
            }
            NodeType::Asinh => {
                let lhs = parent_0.unwrap();
                smallvec!(T::one() / (lhs * lhs + T::one()).sqrt())
            }
            NodeType::Acosh => {
                let lhs = parent_0.unwrap();
                smallvec!(T::one() / (T::sqrt(lhs * lhs - T::one())))
            }
            NodeType::Atanh => {
                let lhs = parent_0.unwrap();
                smallvec!(T::one() / (T::one() - lhs * lhs))
            }
            NodeType::Log => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                let ln_rhs = T::ln(rhs);
                let ln_lhs = T::ln(lhs);
                smallvec!(T::one() / (lhs * ln_rhs), -ln_lhs / (rhs * ln_rhs * ln_rhs))
            }
            NodeType::Ln => {
                let lhs = parent_0.unwrap();
                smallvec![T::one() / lhs]
            }
            NodeType::Sqrt => {
                let lhs = parent_0.unwrap();
                let tmp = if lhs == T::zero() { T::epsilon() } else { lhs };
                smallvec!(T::one() / (T::from(2).unwrap() * T::sqrt(tmp)))
            }
            NodeType::Exp => {
                smallvec!(T::exp(parent_0.unwrap()))
            }
            NodeType::Powf => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                let tmp = if lhs == T::zero() { T::epsilon() } else { lhs };
                smallvec!(
                    rhs * T::powf(lhs, rhs - T::one()),
                    T::powf(lhs, rhs) * T::ln(tmp)
                )
            }
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum NodeIdx {
    Constant,
    Idx(usize),
}

impl NodeIdx {
    #[inline]
    pub fn get_idx(&self) -> usize {
        match self {
            NodeIdx::Constant => unreachable!("cannot get idx from constant."),
            NodeIdx::Idx(idx) => *idx,
        }
    }
}

pub struct GlobalComputationGraph<T: 'static>(&'static ComputationGraph<T>);

impl<T> GlobalComputationGraph<T> {
    #[inline(always)]
    pub(super) fn get() -> GlobalComputationGraph<T> {
        fn build_graph<T>() -> NonNull<()> {
            let graph = ComputationGraph::<T>::new();
            let leaked = Box::leak(Box::new(graph));
            NonNull::from_ref(leaked).cast()
        }

        thread_local! {
            static STORAGE: OnceCell<NonNull<()>> = const { OnceCell::new() };
        }

        let graph_ptr = STORAGE.with(|cell| {
            let ptr = cell.get_or_init(build_graph::<T>);
            ptr.cast::<ComputationGraph<T>>()
        });

        GlobalComputationGraph(unsafe { graph_ptr.as_ref() })
    }

    pub fn reset(&self) {
        self.0.reset();
    }

    pub fn spawn_value(&self, value: T) -> Adr<T>
    where
        T: Clone + Zero,
    {
        self.0.spawn_variable(value)
    }

    pub fn num_nodes(&self) -> usize {
        *self.0.add_idx.read().unwrap()
    }

    #[inline(always)]
    pub fn add_node(
        &self,
        node_type: NodeType,
        value: T,
        parent_0: Option<T>,
        parent_1: Option<T>,
        parent_0_idx: Option<NodeIdx>,
        parent_1_idx: Option<NodeIdx>,
    ) -> Adr<T>
    where
        T: Clone + Zero,
    {
        self.0.add_node(
            node_type,
            value,
            parent_0,
            parent_1,
            parent_0_idx,
            parent_1_idx,
        )
    }

    #[inline(always)]
    pub fn add_unary(&self, parent: Adr<T>, node_type: NodeType, value: T) -> Adr<T>
    where
        T: Clone + Zero,
    {
        self.0.add_node(
            node_type,
            value,
            Some(parent.number),
            None,
            Some(parent.node_idx),
            None,
        )
    }

    pub fn add_binary(&self, a: Adr<T>, b: Adr<T>, node_type: NodeType, value: T) -> Adr<T>
    where
        T: Clone + Zero,
    {
        self.0.add_node(
            node_type,
            value,
            Some(a.number),
            Some(b.number),
            Some(a.node_idx),
            Some(b.node_idx),
        )
    }

    pub fn get_backwards_mode_grad(&self, node_idx: NodeIdx) -> Derivatives<'_, T>
    where
        T: Float + AddAssign,
    {
        self.0.get_backwards_mode_grad(node_idx)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_for {
    (impl $trait:ident { $fn_name:ident $node_type:expr }) => {
        impl<T> $trait for Adr<T>
        where
            T: $trait<Output = T> + Clone + Zero + 'static,
        {
            type Output = Self;

            fn $fn_name(self, rhs: Self) -> Self::Output {
                GlobalComputationGraph::get().add_node(
                    $node_type,
                    T::$fn_name(self.number.clone(), rhs.number.clone()),
                    Some(self.number),
                    Some(rhs.number),
                    Some(self.node_idx),
                    Some(rhs.node_idx),
                )
            }
        }
    };
}

macro_rules! impl_assign_op_for {
    (impl $trait:ident[$base_trait:ident] { $fn_name:ident $fn_assign_name:ident }) => {
        impl<T: $base_trait<Output = T> + Clone + Zero + 'static> $trait for Adr<T> {
            fn $fn_assign_name(&mut self, rhs: Self) {
                *self = Self::$fn_name(self.clone(), rhs)
            }
        }
    };
}

impl_op_for!(impl Add { add NodeType::Add });
impl_op_for!(impl Mul { mul NodeType::Mul });
impl_op_for!(impl Sub { sub NodeType::Sub });
impl_op_for!(impl Div { div NodeType::Div });

impl_assign_op_for!(impl AddAssign[Add] { add add_assign });
impl_assign_op_for!(impl SubAssign[Sub] { sub sub_assign });
impl_assign_op_for!(impl MulAssign[Mul] { mul mul_assign });
impl_assign_op_for!(impl DivAssign[Div] { div div_assign });
impl_assign_op_for!(impl RemAssign[Rem] { rem rem_assign });

impl<T> Neg for Adr<T>
where
    T: Neg<Output = T> + Zero + Clone + 'static,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        GlobalComputationGraph::get().add_node(
            NodeType::Neg,
            T::neg(self.number.clone()),
            Some(self.number),
            None,
            Some(self.node_idx),
            None,
        )
    }
}

impl<T> Rem<Self> for Adr<T> {
    type Output = Self;

    #[inline]
    fn rem(self, _: Self) -> Self::Output {
        todo!()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T: ToPrimitive> ToPrimitive for Adr<T> {
    fn to_i64(&self) -> Option<i64> {
        self.number.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.number.to_u64()
    }
}

impl<N: NumCast> NumCast for Adr<N> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        Some(Adr::constant(N::from(n)?))
    }
}

macro_rules! forward_to_t {
    ($fn_name:ident$(())?) => {
        fn $fn_name() -> Self {
            Self::constant(T::$fn_name())
        }
    };
}

macro_rules! forward_to_number {
    ($fn_name:ident $ret:ty) => {
        fn $fn_name(self) -> $ret {
            T::$fn_name(self.number)
        }
    };
}

macro_rules! unary {
    ($fn_name:ident $node_type:expr) => {
        fn $fn_name(self) -> Self {
            GlobalComputationGraph::get().add_unary(self, $node_type, T::$fn_name(self.number))
        }
    };
}

macro_rules! binary {
    ($fn_name:ident $node_type:expr) => {
        fn $fn_name(self, rhs: Self) -> Self {
            GlobalComputationGraph::get().add_binary(
                self,
                rhs,
                $node_type,
                T::$fn_name(self.number, rhs.number),
            )
        }
    };
}

impl<T: Float + 'static> Float for Adr<T> {
    forward_to_t!(nan);
    forward_to_t!(infinity);
    forward_to_t!(neg_infinity);
    forward_to_t!(neg_zero());
    forward_to_t!(min_value());
    forward_to_t!(min_positive_value());
    forward_to_t!(max_value());

    forward_to_number!(is_nan bool);
    forward_to_number!(is_infinite bool);
    forward_to_number!(is_finite bool);
    forward_to_number!(is_normal bool);
    forward_to_number!(classify FpCategory);
    forward_to_number!(is_sign_positive bool);
    forward_to_number!(is_sign_negative bool);

    unary!(floor NodeType::Floor);
    unary!(ceil NodeType::Ceil);
    unary!(round NodeType::Round);
    unary!(trunc NodeType::Trunc);
    unary!(fract NodeType::Fract);
    unary!(abs NodeType::Abs);
    unary!(signum NodeType::Signum);
    unary!(sin NodeType::Sin);
    unary!(cos NodeType::Cos);
    unary!(tan NodeType::Tan);
    unary!(asin NodeType::Asin);
    unary!(acos NodeType::Acos);
    unary!(atan NodeType::Atan);
    unary!(sinh NodeType::Sinh);
    unary!(cosh NodeType::Cosh);
    unary!(tanh NodeType::Tanh);
    unary!(asinh NodeType::Asinh);
    unary!(acosh NodeType::Acosh);
    unary!(atanh NodeType::Atanh);
    unary!(exp NodeType::Exp);
    unary!(sqrt NodeType::Sqrt);
    unary!(ln NodeType::Ln);

    binary!(min NodeType::Min);
    binary!(max NodeType::Max);
    binary!(powf NodeType::Powf);
    binary!(log NodeType::Log);

    fn mul_add(self, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn recip(self) -> Self {
        todo!()
    }

    fn powi(self, _n: i32) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        todo!()
    }

    fn abs_sub(self, _other: Self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn hypot(self, _other: Self) -> Self {
        todo!()
    }

    fn atan2(self, _other: Self) -> Self {
        todo!()
    }

    fn sin_cos(self) -> (Self, Self) {
        todo!()
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    fn ln_1p(self) -> Self {
        todo!()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.number.integer_decode()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T: PartialEq> PartialEq for Adr<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<T: PartialOrd> PartialOrd for Adr<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<T: Display> Display for Adr<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.number)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T: Zero + Add<Output = T> + Clone + 'static> Zero for Adr<T> {
    #[inline]
    fn zero() -> Self {
        Self::constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl<T: One + Mul<Output = T> + Zero + Clone + 'static> One for Adr<T> {
    #[inline]
    fn one() -> Self {
        Self::constant(T::one())
    }
}

impl<T: Num + 'static> Num for Adr<T>
where
    T: Zero + One + Clone + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::constant(T::from_str_radix(str, radix)?))
    }
}

impl<T: FromPrimitive> FromPrimitive for Adr<T> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(T::from_i64(n)?))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(T::from_u64(n)?))
    }
}

impl<T> Indexed for Adr<T> {
    fn index(&self) -> usize {
        self.node_idx.get_idx()
    }
}

#[cfg(test)]
mod tests {
    use crate::differentiation::{Adr, AD};
    use ndarray_linalg::aclose;
    use num_traits::real::Real;
    use num_traits::NumCast;
    use std::ops::{Add, Mul, Sub};

    fn k<T>(x: T, y: T) -> T
    where
        T: NumCast + Add<Output = T> + Mul<Output = T> + Copy + Sub<Output = T>,
    {
        let three = T::from(3).unwrap();

        x + three * y - x * x * y
    }

    #[test]
    fn test_gradient() {
        let x = Adr::variable(1.0);
        let y = Adr::variable(2.0);
        let z = k(x, y);

        aclose(z.number, k(1.0, 2.0), 1e-15);
        z.with_derivatives(|ds| {
            aclose(ds[&x], -3.0, 1e-15);
            aclose(ds[&y], 2.0, 1e-15);
        });
    }

    fn f<T: Real>(x: T, y: T) -> T {
        x.sin() * y.cos()
    }

    fn g<T: Real>(x: T) -> T {
        (x + T::from(2).unwrap()) * x.ln()
    }

    #[test]
    fn test_simple_derivative() {
        let x = Adr::variable(0.4);

        let y = g(x);
        aclose(y.number, -2.199097756497972, 1e-15);

        y.with_derivatives(|ds| {
            aclose(ds[&x], 5.083709268125845, 1e-15);
        });
    }

    #[test]
    fn test_compound_derivative() {
        let x = Adr::variable(0.4);
        let y = Adr::variable(0.7);

        let z = f(x, y);
        aclose(z.number, 0.2978435767000479, 1e-15);

        z.with_derivatives(|ds| {
            aclose(ds[&x], 0.7044663052755917, 1e-15);
            aclose(ds[&y], -0.2508701838500143, 1e-15);
        });
    }
}
